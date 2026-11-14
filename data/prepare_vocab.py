"""Phase 1: build the symptom vocabulary (knowledge/symptom_vocab.csv).

Purpose: give the extractor a controlled list of symptom names so that "stomach
ache", "pet dard" and "belly pain" all become `abdominal_pain`.

IMPORTANT: this uses the Kaggle disease datasets ONLY for symptom NAMES and a
weak "symptom burden" weight. The disease labels are never used for triage.

Usage (from the project root):
    python data/prepare_vocab.py
    python data/prepare_vocab.py --severity data/raw/Symptom-severity.csv --dataset data/raw/dataset.csv

Download the Kaggle files from
https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
and put Symptom-severity.csv and dataset.csv in data/raw/ . Without them the
script still works from the curated synonyms only.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

# Kaggle names -> our canonical names (all other Kaggle names are kept as they are).
KAGGLE_ALIASES = {
    "high_fever": "fever", "mild_fever": "fever", "stomach_pain": "abdominal_pain",
    "burning_micturition": "burning_urination", "skin_rash": "rash",
    "blurred_and_distorted_vision": "blurred_vision", "altered_sensorium": "confusion",
    "blood_in_sputum": "coughing_blood", "bloody_stool": "blood_in_stool",
    "diarrhoea": "diarrhoea", "breathlessness": "breathlessness", "dizziness": "dizziness",
    "weakness_of_one_body_side": "weakness_of_one_body_side", "slurred_speech": "slurred_speech",
    "continuous_sneezing": "sneezing", "spotting_urination": "blood_in_urine",
}
# Symptoms related to emergency screening. They are kept in the vocabulary but flagged.
RED_FLAG_RELATED = {
    "chest_pain", "breathlessness", "coma", "confusion", "slurred_speech", "weakness_of_one_body_side",
    "coughing_blood", "blood_in_stool", "vomiting_blood", "palpitations", "fainting", "seizure",
    "heavy_bleeding", "loss_of_balance", "unsteadiness",
}


def normalise_name(name: str) -> str:
    """'foul_smell_of urine' -> 'foul_smell_of_urine'."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def load_kaggle_severity(path: Path) -> dict[str, float]:
    weights: dict[str, float] = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2 or not row[0].strip():
                continue
            name = KAGGLE_ALIASES.get(normalise_name(row[0]), normalise_name(row[0]))
            try:
                weight = float(row[1])
            except ValueError:
                continue
            weights[name] = max(weight, weights.get(name, 0))
    return weights


def dataset_report(path: Path) -> str:
    """Counts rows and exact duplicates: tells you how 'templated' the data is."""
    import pandas as pd

    df = pd.read_csv(path)
    symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]
    sets = df[symptom_cols].apply(
        lambda r: tuple(sorted(normalise_name(v) for v in r if isinstance(v, str) and v.strip())), axis=1)
    lines = [
        f"File: {path.name}",
        f"Rows: {len(df)}",
        f"Disease labels: {df.iloc[:, 0].nunique()}",
        f"Exact duplicate rows: {int(df.duplicated().sum())}",
        f"Distinct symptom sets: {sets.nunique()}",
        "Reading: many duplicates / few distinct sets mean the data is templated, "
        "so it must NOT be used to claim any medical accuracy.",
    ]
    return "\n".join(lines)


def build(severity_path: Path | None, dataset_path: Path | None) -> None:
    vocab: dict[str, dict] = {}

    def entry(name: str) -> dict:
        return vocab.setdefault(name, {"weight": "", "synonyms": set(), "sources": set()})

    synonyms = json.loads((config.KNOWLEDGE_DIR / "synonyms.json").read_text(encoding="utf-8"))
    for canonical, syns in synonyms.items():
        if canonical.startswith("_"):
            continue
        e = entry(normalise_name(canonical))
        e["synonyms"].update(s.lower().strip() for s in syns)
        e["sources"].add("curated")

    if severity_path and severity_path.exists():
        for name, weight in load_kaggle_severity(severity_path).items():
            e = entry(name)
            e["weight"] = max(weight, float(e["weight"] or 0))
            e["synonyms"].add(name.replace("_", " "))
            e["sources"].add("kaggle")
        print(f"Merged Kaggle severity file: {severity_path}")
    else:
        print("Kaggle severity file not found: using curated synonyms only "
              "(download Symptom-severity.csv into data/raw/ for the full vocabulary).")

    out_path = config.KNOWLEDGE_DIR / "symptom_vocab.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["symptom", "severity_weight", "red_flag_related", "source", "synonyms"])
        for name in sorted(vocab):
            e = vocab[name]
            syns = sorted(s for s in e["synonyms"] if s != name.replace("_", " "))
            writer.writerow([name, e["weight"], name in RED_FLAG_RELATED,
                             "+".join(sorted(e["sources"])), "|".join(syns)])
    print(f"Wrote {len(vocab)} symptoms to {out_path}")

    if dataset_path and dataset_path.exists():
        report = dataset_report(dataset_path)
        (config.DATA_DIR / "dataset_report.txt").write_text(report, encoding="utf-8")
        print("\n" + report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--severity", type=Path, default=config.DATA_DIR / "raw" / "Symptom-severity.csv")
    ap.add_argument("--dataset", type=Path, default=config.DATA_DIR / "raw" / "dataset.csv")
    args = ap.parse_args()
    build(args.severity, args.dataset)
