"""Extraction accuracy: does the LLM extractor read the patient's message correctly?  (live LLM)

Each case has hand-annotated gold facts. We score (fact, value) pairs after the extractor's Python
validation, exactly what the agent would store. Safety-relevant error types are counted separately:
  false_denial   the extractor stored 'False' where the patient said it IS present  (dangerous)
  missed_true    a symptom the patient reported was not captured                     (dangerous)
  extra_true     a symptom stored as present that the patient never reported         (over-cautious)
"""
from __future__ import annotations

import json
from typing import Any, Optional

import config
from agent.extractor import extract_into_state
from agent.llm_client import LLM
from agent.state import PatientState

NUM_TOL = {"temperature_c": 0.3, "duration_days": 0.6, "age_years": 0.6}


def load_cases(path=None) -> list[dict]:
    with open(path or config.EVAL_DIR / "extraction_cases.json", encoding="utf-8") as fh:
        return json.load(fh)


def same(key: str, gold: Any, pred: Any) -> bool:
    if isinstance(gold, list) and isinstance(pred, list):
        return set(gold) == set(pred)
    if isinstance(gold, (int, float)) and not isinstance(gold, bool) and isinstance(pred, (int, float)):
        return abs(float(gold) - float(pred)) <= NUM_TOL.get(key, 0.01)
    return gold == pred


def score_case(case: dict, applied: dict, dont_know: list[str], complaint: Optional[str]) -> dict:
    """Pure scoring of one extraction (testable without an LLM)."""
    gold = case["gold"]
    tp = fp = fn = 0
    false_denial, missed_true, extra_true = [], [], []
    for key, g in gold.items():
        if key in applied and same(key, g, applied[key]):
            tp += 1
        else:
            fn += 1
            if g is True and applied.get(key) is False:
                false_denial.append(key)
            elif g is True:
                missed_true.append(key)
    for key, p in applied.items():
        if key in gold and same(key, gold[key], p):
            continue
        fp += 1
        if p is True and key not in gold:
            extra_true.append(key)
    dk_ok = all(k in dont_know for k in case.get("dont_know", []))
    want = case.get("complaint")
    complaint_ok = None if want is None else (complaint == want)
    return {"id": case["id"], "tp": tp, "fp": fp, "fn": fn, "false_denial": false_denial, "missed_true": missed_true,
            "extra_true": extra_true, "dont_know_ok": dk_ok, "complaint_ok": complaint_ok,
            "applied": applied, "complaint": complaint}


def run_cases(llm: LLM, cases: Optional[list[dict]] = None) -> list[dict]:
    out = []
    for case in cases or load_cases():
        state = PatientState()
        if case.get("setup", {}).get("complaint"):
            state.complaint = case["setup"]["complaint"]
        res = extract_into_state(state, case["text"], llm, pending_key=case.get("pending_key"))
        out.append(score_case(case, dict(res.applied), list(res.dont_know), state.complaint)
                   | {"error": res.error or None, "fast_path": res.fast_path})
    return out


def summarize(rows: list[dict]) -> dict:
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    comp = [r["complaint_ok"] for r in rows if r["complaint_ok"] is not None]
    return {"cases": len(rows), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "false_denials": sum(len(r["false_denial"]) for r in rows),
            "missed_true": sum(len(r["missed_true"]) for r in rows),
            "extra_true": sum(len(r["extra_true"]) for r in rows),
            "complaint_accuracy": sum(comp) / len(comp) if comp else None, "complaint_n": len(comp),
            "dont_know_ok": sum(r["dont_know_ok"] for r in rows), "errors": sum(1 for r in rows if r.get("error"))}
