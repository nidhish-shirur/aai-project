from __future__ import annotations

from typing import Any

from fpdf import FPDF


def _safe(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\r", " ").replace("\n", " ")
    return (
        text.encode("latin-1", "replace")
        .decode("latin-1")
        .strip()
    )


def _available_width(pdf: FPDF) -> float:
    return pdf.w - pdf.l_margin - pdf.r_margin


def build_clinician_pdf(
    summary: dict[str, Any],
    report: dict[str, Any] | None,
    timeline: list[str],
) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(
        auto=True,
        margin=15,
    )
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(
        0,
        10,
        "TriageGuide Clinical Summary",
        ln=True,
    )

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0,
        7,
        "Educational decision-support prototype - not a diagnosis",
        ln=True,
    )
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Assessment", ln=True)

    pdf.set_font("Helvetica", "", 10)

    fields = [
        ("Patient code", summary.get("patient_code")),
        ("Urgency", summary.get("urgency_label")),
        ("Suggested care", summary.get("suggested_care")),
        ("Timeframe", summary.get("timeframe")),
    ]

    for label, value in fields:
        width = _available_width(pdf)

        pdf.multi_cell(
            width,
            6,
            f"{label}: {_safe(value)}",
        )

    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Clinical summary", ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        _available_width(pdf),
        6,
        "Clinical summary for clinician",
    )

    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Reasons", ln=True)

    pdf.set_font("Helvetica", "", 10)
    width = _available_width(pdf)

    # Reasons
    for reason in summary.get("reasons", []):
        pdf.multi_cell(
            width,
            6,
            f"- {_safe(reason)}",
        )

    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Clinician review", ln=True)

    pdf.set_font("Helvetica", "", 10)
    review_required = summary.get("needs_review", False)

    pdf.multi_cell(
        width,
        6,
        f"Required: {'Yes' if review_required else 'No'}",
    )

    for reason in summary.get("review_reasons", []):
        pdf.multi_cell(
            width,
            6,
            f"- {_safe(reason)}",
        )

    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Symptom timeline", ln=True)

    pdf.set_font("Helvetica", "", 10)

    # Timeline
    for item in timeline:
        pdf.multi_cell(
            width,
            6,
            f"- {_safe(item)}",
        )

    if report:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Workflow trace", ln=True)

        pdf.set_font("Helvetica", "", 9)

        for item in report.get("tool_log", []):
            if isinstance(item, dict):
                tool = item.get("tool") or item.get("node")
                if tool:
                    # Workflow
                    pdf.multi_cell(
                        width,
                        5,
                        f"- {_safe(tool)}",
                    )

    return bytes(pdf.output())