from __future__ import annotations


# --------------------------------------------------------------------------
# Graph definition
# --------------------------------------------------------------------------

NODES: dict[str, str] = {
    "START": '(["Start"])',
    "LANGUAGE": '["Language detection"]',
    "TEXT_FLAGS": '["Red flag detection"]',
    "EMERGENCY": '["Emergency escalation"]',
    "LANG_AGENT": '["Language agent"]',
    "COLLECTION": '["Symptom collection"]',
    "FACT_FLAGS": '["Fact red flag detection"]',
    "QUESTIONS": '["Adaptive questioning"]',
    "WAIT": '["Wait for patient response"]',
    "KNOWLEDGE": '["Medical knowledge"]',
    "RISK": '["Risk assessment"]',
    "REVIEW": '["Safety review"]',
    "CLINICIAN": '["Clinician escalation"]',
    "CARE": '["Care pathway"]',
    "SUMMARY": '["Medical summary"]',
    "MEMORY": '["Memory"]',
    "DONE": '(["End"])',
}

# (source, target, label). The order defines the Mermaid link index,
# which is used to highlight the edges a session actually took.
EDGES: list[tuple[str, str, str | None]] = [
    ("START", "LANGUAGE", None),
    ("LANGUAGE", "TEXT_FLAGS", None),
    ("TEXT_FLAGS", "EMERGENCY", "Emergency"),
    ("TEXT_FLAGS", "LANG_AGENT", "Safe"),
    ("LANG_AGENT", "COLLECTION", None),
    ("COLLECTION", "FACT_FLAGS", None),
    ("FACT_FLAGS", "EMERGENCY", "Emergency"),
    ("FACT_FLAGS", "QUESTIONS", "Safe"),
    ("QUESTIONS", "WAIT", "Missing information"),
    ("QUESTIONS", "KNOWLEDGE", "Complete"),
    ("WAIT", "DONE", None),
    ("KNOWLEDGE", "RISK", None),
    ("RISK", "REVIEW", None),
    ("REVIEW", "CLINICIAN", None),
    ("CLINICIAN", "CARE", None),
    ("CARE", "SUMMARY", None),
    ("SUMMARY", "MEMORY", None),
    ("EMERGENCY", "MEMORY", None),
    ("MEMORY", "DONE", None),
]

# tool_log "tool" name -> Mermaid node ID
TOOL_TO_NODE: dict[str, str] = {
    "language_detection": "LANGUAGE",
    "red_flag_agent": "TEXT_FLAGS",
    "language_agent": "LANG_AGENT",
    "symptom_collection_agent": "COLLECTION",
    "fact_red_flag_agent": "FACT_FLAGS",
    "adaptive_questioning_agent": "QUESTIONS",
    "medical_knowledge_agent": "KNOWLEDGE",
    "risk_assessment_agent": "RISK",
    "safety_agent": "REVIEW",
    "clinician_escalation_agent": "CLINICIAN",
    "care_pathway_agent": "CARE",
    "medical_summary_agent": "SUMMARY",
    "emergency_agent": "EMERGENCY",
}

ALL_NODE_IDS: list[str] = list(NODES)

# Which gate sends the session to emergency escalation.
GATE_TOOLS: dict[str, str] = {
    "red_flag_agent": "TEXT_FLAGS",
    "fact_red_flag_agent": "FACT_FLAGS",
}


def _base_diagram() -> str:
    lines = ["flowchart TD"]

    for node_id, shape in NODES.items():
        lines.append(f"    {node_id}{shape}")

    for src, dst, label in EDGES:
        arrow = f'-->|"{label}"|' if label else "-->"
        lines.append(f"    {src} {arrow} {dst}")

    return "\n".join(lines)


# Kept for code that imports the plain diagram.
WORKFLOW_MERMAID = _base_diagram()


# --------------------------------------------------------------------------
# Execution analysis
# --------------------------------------------------------------------------

def executed_nodes(report: dict | None) -> list[str]:
    """Tool names that ran, in order, without duplicates."""
    if not report:
        return []

    nodes: list[str] = []

    for item in report.get("tool_log", []):
        if not isinstance(item, dict):
            continue

        tool = item.get("tool")

        if isinstance(tool, str) and tool not in nodes:
            nodes.append(tool)

    return nodes


def executed_node_ids(report: dict | None) -> list[str]:
    """Mermaid node IDs for the steps this session actually passed through."""
    if not report:
        return []

    ids: list[str] = ["START"]

    for tool in executed_nodes(report):
        node_id = TOOL_TO_NODE.get(tool)

        if node_id and node_id not in ids:
            ids.append(node_id)

    # A question was asked at least once (a "field" entry in the log).
    asked = any(
        isinstance(item, dict)
        and item.get("tool") == "adaptive_questioning_agent"
        and "field" in item
        for item in report.get("tool_log", [])
    )

    if asked:
        ids.append("WAIT")

    # memory_agent doesn't write a tool_log entry, but it ran if a report exists.
    ids.extend(["MEMORY", "DONE"])

    return ids


def emergency_gate(report: dict | None) -> str | None:
    """Node ID of the red-flag gate that triggered the emergency, if any."""
    if not report:
        return None

    gate = None

    for item in report.get("tool_log", []):
        if not isinstance(item, dict):
            continue

        node_id = GATE_TOOLS.get(item.get("tool", ""))

        if node_id and item.get("triggered"):
            gate = node_id

    return gate


def _edge_active(
    src: str,
    dst: str,
    ids: set[str],
    gate: str | None,
) -> bool:
    if src not in ids or dst not in ids:
        return False

    # Only the gate that fired is credited with the emergency edge.
    if dst == "EMERGENCY":
        return src == gate

    return True


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def workflow_diagram(report: dict | None = None) -> str:
    """Mermaid diagram, with the executed path highlighted when a report is given."""
    lines = [_base_diagram(), ""]

    lines += [
        "classDef executed fill:#dff4e8,stroke:#18794e,stroke-width:2px,color:#102a43",
        "classDef flagged fill:#fff3d6,stroke:#c98a00,stroke-width:3px,color:#7a5200",
        "classDef emergency fill:#fde7e7,stroke:#d64545,stroke-width:3px,color:#b42318",
        "classDef idle fill:#ffffff,stroke:#cbd5e1,stroke-dasharray:4 3,color:#829ab1",
    ]

    if not report:
        return "\n".join(lines)

    ids = executed_node_ids(report)
    id_set = set(ids)
    gate = emergency_gate(report)

    # Amber: steps that raised a flag for the clinician.
    flagged: set[str] = set()

    if "CLINICIAN" in id_set and report.get("needs_clinician_review"):
        flagged.add("CLINICIAN")

    review = report.get("safety_review") or {}

    if "REVIEW" in id_set and isinstance(review, dict) and review.get("raised"):
        flagged.add("REVIEW")

    green = [i for i in ids if i not in flagged and i != "EMERGENCY"]
    idle = [i for i in ALL_NODE_IDS if i not in id_set]

    if green:
        lines.append(f"class {','.join(green)} executed")
    if flagged:
        lines.append(f"class {','.join(sorted(flagged))} flagged")
    if "EMERGENCY" in id_set:
        lines.append("class EMERGENCY emergency")
    if idle:
        lines.append(f"class {','.join(idle)} idle")

    # Edges: green for the path taken, red into emergency, grey otherwise.
    active, emergency_edges, muted = [], [], []

    for index, (src, dst, _label) in enumerate(EDGES):
        if _edge_active(src, dst, id_set, gate):
            (emergency_edges if dst == "EMERGENCY" else active).append(index)
        else:
            muted.append(index)

    if active:
        lines.append(
            f"linkStyle {','.join(map(str, active))} "
            "stroke:#18794e,stroke-width:3px"
        )
    if emergency_edges:
        lines.append(
            f"linkStyle {','.join(map(str, emergency_edges))} "
            "stroke:#d64545,stroke-width:3px"
        )
    if muted:
        lines.append(
            f"linkStyle {','.join(map(str, muted))} "
            "stroke:#cbd5e1,stroke-width:1px"
        )

    return "\n".join(lines)