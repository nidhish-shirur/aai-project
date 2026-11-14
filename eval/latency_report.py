"""Latency report from the sessions saved in your database (run after some chats).

Run: python eval/latency_report.py [path/to/triageguide.db]
Reads every finished session's stored tool log, so it needs no LLM and no network.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from eval.metrics import percentile  # noqa: E402


def collect(reports: list[dict]) -> dict:
    """reports = list of saved report dicts. -> per-purpose and per-turn timings."""
    per_call: dict[str, list[float]] = {}
    per_turn: list[float] = []
    fast = slow_timeouts = rate_hits = 0
    for rep in reports:
        turns: dict[int, float] = {}
        for t in rep.get("tool_log", []):
            if t["tool"] == "extract_facts" and t.get("fast_path"):
                fast += 1
            if t["tool"] == "llm_calls":
                for c in t["calls"]:
                    per_call.setdefault(c["label"], []).append(c["seconds"])
                    rate_hits += c.get("rate_limit_hits", 0) or 0
                    if not c["ok"] and "timed out" in str(c.get("error", "")).lower():
                        slow_timeouts += 1
                turns[t["turn"]] = turns.get(t["turn"], 0) + t.get("total_s", 0)
        per_turn.extend(turns.values())
    def stats(vals: list[float]) -> dict:
        return {"n": len(vals), "median": percentile(vals, 50), "p90": percentile(vals, 90),
                "p95": percentile(vals, 95), "max": max(vals) if vals else None}
    return {"sessions": len(reports), "by_purpose": {k: stats(v) for k, v in sorted(per_call.items())},
            "per_turn_llm_time": stats(per_turn), "fast_path_answers": fast,
            "rate_limit_hits": rate_hits, "timeouts": slow_timeouts}


def load_reports(db_path: Optional[str] = None) -> list[dict]:
    con = sqlite3.connect(db_path or str(config.DB_PATH))
    rows = con.execute("SELECT report_json FROM sessions WHERE report_json IS NOT NULL").fetchall()
    con.close()
    return [json.loads(r[0]) for r in rows]


def main() -> None:
    db = sys.argv[1] if len(sys.argv) > 1 else None
    reports = [r for r in load_reports(db) if r.get("llm_used")]
    if not reports:
        print("No finished sessions that used the LLM were found. Chat with the AI model on, then run this again.")
        return
    d = collect(reports)
    print(f"Sessions with the LLM: {d['sessions']}   fast-path answers (no LLM call): {d['fast_path_answers']}"
          f"   rate-limit hits: {d['rate_limit_hits']}   timeouts: {d['timeouts']}")
    print(f"{'purpose':<10}{'calls':>6}{'median':>9}{'p90':>8}{'p95':>8}{'max':>8}   (seconds)")
    for name, s in d["by_purpose"].items():
        print(f"{name:<10}{s['n']:>6}{s['median']:>9.2f}{s['p90']:>8.2f}{s['p95']:>8.2f}{s['max']:>8.2f}")
    t = d["per_turn_llm_time"]
    print(f"{'whole turn':<10}{t['n']:>6}{t['median']:>9.2f}{t['p90']:>8.2f}{t['p95']:>8.2f}{t['max']:>8.2f}"
          "   <- time the patient waits per message (target: median < 4 s)")


if __name__ == "__main__":
    main()
