"""Writes eval/review_sheet.csv for a CLINICIAN to label the held-out cases independently.

The sheet shows each case in plain English and deliberately does NOT show the AI-drafted label or the
system's output, so the clinician's judgement is independent. Give them the sheet, ask them to fill the
'clinician_label' column with one of: low / moderate / urgent / emergency (definitions are in the header rows),
then run:  python eval/run_eval.py --reviewed path/to/filled_sheet.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

DEFINITIONS = [
    "LABELS: emergency = call emergency services / go to the emergency department right now",
    "LABELS: urgent = be seen at an emergency department or urgent care today, within hours",
    "LABELS: moderate = see a doctor within 1 to 3 days",
    "LABELS: low = self-care and monitoring at home",
]


def plain_facts(facts: dict) -> str:
    parts = []
    for k, v in facts.items():
        name = k.replace("_", " ")
        if v is True:
            parts.append(f"has {name}")
        elif v is False:
            parts.append(f"no {name}")
        elif isinstance(v, list):
            parts.append("long-term conditions: " + (", ".join(x.replace('_', ' ') for x in v) or "none"))
        else:
            parts.append(f"{name}: {v}")
    return "; ".join(parts)


def main() -> None:
    cases = json.load(open(config.EVAL_DIR / "heldout_cases.json", encoding="utf-8"))
    out = config.EVAL_DIR / "review_sheet.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        for line in DEFINITIONS:
            w.writerow([line])
        w.writerow([])
        w.writerow(["id", "what the patient says", "additional facts", "clinician_label", "comment"])
        for c in cases:
            w.writerow([c["id"], c["text"], plain_facts(c["facts"]), "", ""])
    print(f"Wrote {out} ({len(cases)} cases). Give it to a doctor or nurse to fill in 'clinician_label'.")


if __name__ == "__main__":
    main()
