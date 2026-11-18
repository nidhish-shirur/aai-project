from __future__ import annotations


WORKFLOW_MERMAID = """
flowchart TD
    START(["Start"]) --> LANGUAGE["Language detection"]
    LANGUAGE --> TEXT_FLAGS["Red flag detection"]

    TEXT_FLAGS -->|"Emergency"| EMERGENCY["Emergency escalation"]
    TEXT_FLAGS -->|"Safe"| LANG_AGENT["Language agent"]

    LANG_AGENT --> COLLECTION["Symptom collection"]
    COLLECTION --> FACT_FLAGS["Fact red flag detection"]

    FACT_FLAGS -->|"Emergency"| EMERGENCY
    FACT_FLAGS -->|"Safe"| QUESTIONS["Adaptive questioning"]

    QUESTIONS -->|"Missing information"| WAIT["Wait for patient response"]
    QUESTIONS -->|"Complete"| KNOWLEDGE["Medical knowledge"]
    WAIT --> DONE(["End"])

    KNOWLEDGE --> RISK["Risk assessment"]
    RISK --> REVIEW["Safety review"]
    REVIEW --> CLINICIAN["Clinician escalation"]
    CLINICIAN --> CARE["Care pathway"]
    CARE --> SUMMARY["Medical summary"]

    SUMMARY --> MEMORY["Memory"]
    EMERGENCY --> MEMORY
    MEMORY --> DONE
"""

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

ALL_NODE_IDS: list[str] = [
    "START", "LANGUAGE", "TEXT_FLAGS", "LANG_AGENT", "COLLECTION",
    "FACT_FLAGS", "QUESTIONS", "WAIT", "KNOWLEDGE", "RISK", "REVIEW",
    "CLINICIAN", "CARE", "SUMMARY", "EMERGENCY", "MEMORY", "DONE",
]


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


def workflow_diagram(report: dict | None = None) -> str:
    """Mermaid diagram, with the executed path highlighted when a report is given."""
    lines = [WORKFLOW_MERMAID.strip(), ""]

    lines += [
        "classDef executed fill:#dff4e8,stroke:#18794e,stroke-width:2px,color:#102a43",
        "classDef emergency fill:#fde7e7,stroke:#d64545,stroke-width:3px,color:#b42318",
        "classDef idle fill:#ffffff,stroke:#cbd5e1,stroke-dasharray:4 3,color:#829ab1",
    ]

    if report:
        ids = executed_node_ids(report)
        emergency = "EMERGENCY" in ids
        green = [i for i in ids if i != "EMERGENCY"]
        idle = [i for i in ALL_NODE_IDS if i not in ids]

        if green:
            lines.append(f"class {','.join(green)} executed")
        if emergency:
            lines.append("class EMERGENCY emergency")
        if idle:
            lines.append(f"class {','.join(idle)} idle")

    return "\n".join(lines)