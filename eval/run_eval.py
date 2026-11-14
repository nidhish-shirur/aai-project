"""Phase 8 evaluation runner.

Offline (no key):   python eval/run_eval.py
With your Groq key: python eval/run_eval.py --live              (adds LLM-only baseline + rules+AI review)
                    python eval/run_eval.py --live --extraction (adds extractor accuracy, 30 messages)
                    python eval/run_eval.py --live --full       (adds the complete agent with LLM; many calls, slow)
Clinician labels:   python eval/run_eval.py --reviewed eval/review_sheet_filled.csv

Writes eval/results/results_<tag>.json and RESULTS_<tag>.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from eval import extraction_eval  # noqa: E402
from eval.ablations import ABLATIONS, ablation  # noqa: E402
from eval.baselines import llm_only_urgency  # noqa: E402
from eval.metrics import LEVELS, by_group, cohen_kappa_nominal, summarize  # noqa: E402
from eval.sim_patient import run_case  # noqa: E402

CASES = config.EVAL_DIR / "heldout_cases.json"
CHECKSUM = config.EVAL_DIR / "heldout_cases.sha256"
RESULTS = config.EVAL_DIR / "results"


# ------------------------------------------------------------------ provenance
def sha256_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def system_fingerprint() -> str:
    """Hash of everything that decides urgency: knowledge files + engine code. Changes when the system changes."""
    files = [p for p in config.KNOWLEDGE_DIR.rglob("*.json")] + list((config.BASE_DIR / "engine").glob("*.py")) \
        + [config.BASE_DIR / "agent" / "orchestrator.py", config.BASE_DIR / "agent" / "extractor.py"]
    return sha256_files(files)[:16]


def labels_frozen_ok() -> bool:
    if not CHECKSUM.exists():
        return False
    return hashlib.sha256(CASES.read_bytes()).hexdigest() == CHECKSUM.read_text().strip()


def load_cases() -> list[dict]:
    return json.loads(CASES.read_text(encoding="utf-8"))


def load_reviewed(path: str) -> dict[str, str]:
    """clinician labels from the filled CSV: {case id: label}. Rows with an empty/invalid label are skipped."""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.reader(fh):
            if len(row) >= 4 and row[0] and row[3].strip().lower() in LEVELS:
                out[row[0].strip()] = row[3].strip().lower()
    return out


# ------------------------------------------------------------------ systems
def run_rules(cases: list[dict]) -> list[dict]:
    rows = []
    for c in cases:
        r = run_case(c, llm=None)
        res = r["result"] or {}
        rows.append({"id": c["id"], "group": c["group"], "gold": c["gold"], "pred": r["pred"] or "low",
                     "finished": r["done"], "turns": r["turns"], "protocol": res.get("protocol_id"),
                     "reasons": res.get("reasons", []), "rules": res.get("matched_rules", []),
                     "confidence": res.get("confidence"), "_run": r})
    return rows


def run_baseline(cases: list[dict], base_rows: list[dict], llm, pause: float) -> tuple[list[dict], int]:
    rows, invalid = [], 0
    for c, b in zip(cases, base_rows):
        level, err = llm_only_urgency(llm, b["_run"]["transcript"])
        if level is None:
            invalid += 1
            print(f"   baseline {c['id']}: {err}")
            continue
        rows.append({"id": c["id"], "group": c["group"], "gold": c["gold"], "pred": level})
        time.sleep(pause)
    return rows, invalid


def run_rules_plus_review(cases: list[dict], base_rows: list[dict], llm, pause: float) -> list[dict]:
    from agent.safety_review import review_and_apply
    from engine.triage_rules import compute_urgency
    rows = []
    for c, b in zip(cases, base_rows):
        state = b["_run"]["state"]
        result = compute_urgency(state)
        outcome = review_and_apply(llm, state, result)
        rows.append({"id": c["id"], "group": c["group"], "gold": c["gold"], "pred": result.urgency,
                     "raised": outcome.raised, "review_error": outcome.error or None})
        time.sleep(pause)
    return rows


def run_full_agent(cases: list[dict], llm, limit: int | None) -> list[dict]:
    rows = []
    for c in cases[:limit]:
        r = run_case(c, llm=llm, safety_review=True)
        rows.append({"id": c["id"], "group": c["group"], "gold": c["gold"], "pred": r["pred"] or "low",
                     "turns": r["turns"], "llm_seconds": r["latency"]["llm_seconds"],
                     "llm_calls": r["latency"]["llm_calls"]})
        print(f"   full agent {c['id']}: gold={c['gold']} pred={r['pred']} turns={r['turns']} "
              f"llm={r['latency']['llm_seconds']}s")
    return rows


# ------------------------------------------------------------------ markdown
def pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def system_row(name: str, s: dict) -> list:
    lo, hi = s["exact_ci"]
    return [name, s["n"], f"{pct(s['exact'])} ({lo:.0%}-{hi:.0%})", pct(s["within_one"]), pct(s["under_triage"]),
            pct(s["over_triage"]), pct(s["emergency_recall"]), pct(s["urgent_or_higher_recall"]),
            s["dangerous_under_triage"], "n/a" if s["weighted_kappa"] is None else f"{s['weighted_kappa']:.2f}"]


MAIN_HEADERS = ["System", "n", "Exact (95% CI)", "Within 1", "Under-triage", "Over-triage", "Emergency recall",
                "Urgent+ recall", "Dangerous misses", "Weighted kappa"]


def confusion_md(c: list[list[int]]) -> str:
    return md_table(["gold \\ predicted", *LEVELS], [[LEVELS[i], *c[i]] for i in range(4)])


def failures_md(rows: list[dict]) -> str:
    bad = [r for r in rows if r["pred"] != r["gold"]]
    if not bad:
        return "No disagreements."
    out = []
    for r in bad:
        kind = "UNDER-triage" if LEVELS.index(r["pred"]) < LEVELS.index(r["gold"]) else "over-triage"
        out.append([r["id"], r["group"], r["gold"], r["pred"], kind, r.get("protocol"),
                    "; ".join(r.get("reasons", []))[:110]])
    return md_table(["id", "group", "gold", "predicted", "kind", "protocol used", "why the system said so"], out)


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="LLM-only baseline and rules+AI review (needs key)")
    ap.add_argument("--full", action="store_true", help="complete agent with LLM (many calls)")
    ap.add_argument("--extraction", action="store_true", help="extractor accuracy on 30 annotated messages (needs key)")
    ap.add_argument("--reviewed", help="CSV with clinician labels (from export_review_sheet.py)")
    ap.add_argument("--limit", type=int, help="only the first N cases for --full")
    ap.add_argument("--pause", type=float, default=0.3, help="seconds between live calls")
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--allow-edited", action="store_true", help="do not stop if the case file changed since freezing")
    args = ap.parse_args()

    if not labels_frozen_ok() and not args.allow_edited:
        sys.exit("heldout_cases.json differs from the frozen checksum (heldout_cases.sha256): the labels were edited "
                 "after freezing. Restore the file, or pass --allow-edited and say so in your report.")
    cases = load_cases()
    gold_source = "AI-drafted labels"
    reviewed = {}
    agreement = None
    if args.reviewed:
        reviewed = load_reviewed(args.reviewed)
        ids = [c["id"] for c in cases if c["id"] in reviewed]
        agreement = {"n": len(ids), "exact": sum(reviewed[i] == next(c["gold"] for c in cases if c["id"] == i)
                                                  for i in ids) / len(ids) if ids else None,
                     "kappa": cohen_kappa_nominal([next(c["gold"] for c in cases if c["id"] == i) for i in ids],
                                                  [reviewed[i] for i in ids])}
        for c in cases:
            c["ai_gold"] = c["gold"]
            c["gold"] = reviewed.get(c["id"], c["gold"])
        gold_source = f"clinician labels for {len(ids)} of {len(cases)} cases (rest AI-drafted)"

    print(f"{len(cases)} held-out cases | gold: {gold_source} | labels frozen: {labels_frozen_ok()} | "
          f"system {system_fingerprint()}")

    print("\n[A] Rule-based system, no LLM (full agent pipeline, simulated patient)...")
    base = run_rules(cases)
    systems: dict[str, list[dict]] = {"A. Rules + gate (no LLM)": base}

    ablation_rows = {}
    for name in ABLATIONS:
        if name == "full_system":
            continue
        with ablation(name):
            rows = run_rules(cases)
        ablation_rows[name] = rows

    llm = None
    notes = []
    invalid = 0
    if args.live or args.full or args.extraction:
        from agent.llm_client import get_llm
        try:
            llm = get_llm()
            llm.complete_json("Reply with JSON.", '{"ping": true}')
        except Exception as exc:
            sys.exit(f"Cannot reach the LLM: {exc}\nFix the connection first (python scripts/test_llm.py).")
    if llm and args.live:
        print("\n[B] LLM-only baseline (same transcript, no rules)...")
        rows, invalid = run_baseline(cases, base, llm, args.pause)
        systems["B. LLM-only baseline"] = rows
        print("\n[C] Rules + AI safety review (escalate-only)...")
        rows = run_rules_plus_review(cases, base, llm, args.pause)
        systems["C. Rules + AI review"] = rows
    if llm and args.full:
        print("\n[D] Complete agent with LLM (extractor + planner + review)...")
        systems["D. Full agent with LLM"] = run_full_agent(cases, llm, args.limit)

    extraction = None
    if llm and args.extraction:
        print("\n[X] Extraction accuracy...")
        rows = extraction_eval.run_cases(llm)
        extraction = {"summary": extraction_eval.summarize(rows), "rows": rows}

    # ---------------- report
    sums = {name: summarize([(r["gold"], r["pred"]) for r in rows]) for name, rows in systems.items()}
    abl_sums = {name: summarize([(r["gold"], r["pred"]) for r in rows]) for name, rows in ablation_rows.items()}
    full_pred = {r["id"]: r["pred"] for r in base}

    md = [f"# TriageGuide evaluation ({args.tag})", "",
          f"* Date: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC | cases: {len(cases)} | gold labels: **{gold_source}**",
          f"* Case file frozen (checksum matches): **{labels_frozen_ok()}** | system fingerprint: `{system_fingerprint()}`",
          "* These are small-sample, student-built results. Intervals are 95% Wilson intervals; do not over-read differences of a few cases.",
          ""]
    if agreement:
        md += [f"**AI-draft vs clinician agreement:** exact {pct(agreement['exact'])} on {agreement['n']} cases, "
               f"Cohen's kappa {agreement['kappa'] if agreement['kappa'] is None else round(agreement['kappa'], 2)}", ""]
    md += ["## 1. Main comparison", md_table(MAIN_HEADERS, [system_row(n, s) for n, s in sums.items()]), ""]
    if invalid:
        md += [f"_LLM-only baseline: {invalid} reply(ies) were invalid and are excluded._", ""]
    md += ["Dangerous miss = an emergency sent to moderate/low, or an urgent case sent to low.", "",
           "## 2. Confusion matrix of system A (rows gold, columns predicted)", confusion_md(sums[next(iter(sums))]["confusion"]), ""]
    for name, s in sums.items():
        if name != next(iter(sums)):
            md += [f"### Confusion matrix: {name}", confusion_md(s["confusion"]), ""]
    groups = by_group(base)
    md += ["## 3. System A by case group",
           md_table(["group", "n", "exact", "under-triage", "over-triage"],
                    [[g, s["n"], pct(s["exact"]), pct(s["under_triage"]), pct(s["over_triage"])] for g, s in groups.items()]), ""]
    md += ["## 4. Ablations (system A with one component switched off)",
           md_table(["Variant", "Exact", "Under-triage", "Over-triage", "Emergency recall", "Dangerous misses", "Cases that changed"],
                    [["full system", pct(sums[next(iter(sums))]["exact"]), pct(sums[next(iter(sums))]["under_triage"]),
                      pct(sums[next(iter(sums))]["over_triage"]), pct(sums[next(iter(sums))]["emergency_recall"]),
                      sums[next(iter(sums))]["dangerous_under_triage"], "-"]] +
                    [[f"{n}: {ABLATIONS[n]}", pct(s["exact"]), pct(s["under_triage"]), pct(s["over_triage"]),
                      pct(s["emergency_recall"]), s["dangerous_under_triage"],
                      ", ".join(f"{r['id']} ({full_pred[r['id']]}->{r['pred']})" for r in ablation_rows[n]
                                if r["pred"] != full_pred[r["id"]]) or "none"]
                     for n, s in abl_sums.items()]), "",
           "A component that changes no case on this set is simply not exercised by it (it may still matter in real use).", ""]
    md += ["## 5. Where system A disagreed with the gold label (failure analysis)", failures_md(base), ""]
    if "C. Rules + AI review" in systems:
        raised = [r for r in systems["C. Rules + AI review"] if r.get("raised")]
        errs = sum(1 for r in systems["C. Rules + AI review"] if r.get("review_error"))
        md += ["## 6. What the AI review did",
               f"It raised the urgency on {len(raised)} of {len(cases)} cases"
               + (f" ({', '.join(r['id'] + ': ' + r['gold'] + '->' + r['pred'] for r in raised)})" if raised else "")
               + (f"; {errs} review call(s) failed and were skipped." if errs else "."), ""]
    if extraction:
        e = extraction["summary"]
        md += ["## 7. Extraction accuracy (live LLM, 30 annotated messages)",
               md_table(["metric", "value"], [["precision", pct(e["precision"])], ["recall", pct(e["recall"])], ["F1", pct(e["f1"])],
                                              ["false denials (said no, patient said yes)", e["false_denials"]],
                                              ["missed reported symptoms", e["missed_true"]],
                                              ["extra symptoms not reported", e["extra_true"]],
                                              ["complaint accuracy", f"{pct(e['complaint_accuracy'])} of {e['complaint_n']}"],
                                              ["LLM errors (fallback used)", e["errors"]]]), ""]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"RESULTS_{args.tag}.md").write_text("\n".join(md), encoding="utf-8")
    payload = {"tag": args.tag, "date": datetime.now(timezone.utc).isoformat(), "fingerprint": system_fingerprint(),
               "labels_frozen": labels_frozen_ok(), "gold_source": gold_source, "agreement": agreement,
               "systems": {n: {k: v for k, v in s.items()} for n, s in sums.items()},
               "ablations": abl_sums,
               "cases": [{k: v for k, v in r.items() if k != "_run"} for r in base],
               "extraction": extraction["summary"] if extraction else None}
    (RESULTS / f"results_{args.tag}.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")

    print("\n" + md_table(MAIN_HEADERS, [system_row(n, s) for n, s in sums.items()]))
    print("\nConfusion matrix, system A (rows gold, columns predicted):\n" + confusion_md(sums[next(iter(sums))]["confusion"]))
    bad = [r for r in base if r["pred"] != r["gold"]]
    print("\nSystem A disagreements: " + ", ".join(f"{r['id']}({r['gold']}->{r['pred']})" for r in bad))
    if extraction:
        print("Extraction:", extraction["summary"])
    print(f"\nFull report: {RESULTS / f'RESULTS_{args.tag}.md'}")


if __name__ == "__main__":
    main()
