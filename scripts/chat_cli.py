"""Phase 5: chat with the triage agent in the terminal (LLM orchestrator + safety core).

Run from the project root:   python scripts/chat_cli.py
Options:  --no-llm   deterministic mode (same as the fallback, but through the agent)
          --trace    print the tool calls the agent made after every turn (great for the viva)
Type "quit" to stop.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.llm_client import LLMError, get_llm  # noqa: E402
from agent.orchestrator import AgentTurn, TriageAgent  # noqa: E402
from memory.db import Memory  # noqa: E402

BAR = "=" * 64


def show_trace(turn: AgentTurn) -> None:
    for t in turn.tool_log:
        if t["tool"] == "llm_calls":
            for c in t["calls"]:
                extra = []
                if c.get("attempts", 1) > 1:
                    extra.append(f"attempts={c['attempts']}")
                if c.get("rate_limit_hits"):
                    extra.append(f"RATE-LIMITED x{c['rate_limit_hits']}, waited {c.get('sleep_s', 0)}s")
                if c.get("completion_tokens"):
                    extra.append(f"{c['completion_tokens']} output tokens")
                if c.get("json_retries"):
                    extra.append(f"json_retries={c['json_retries']}")
                if not c["ok"]:
                    extra.append("FAILED: " + str(c["error"])[:60])
                print(f"   [llm] {c['label']:8s} {c['seconds']:5.1f}s  " + ", ".join(extra))
            continue
        detail = {k: v for k, v in t.items() if k not in ("turn", "tool") and v not in (None, [], "")}
        print(f"   [tool] {t['tool']}  {detail}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    llm = None
    if not args.no_llm:
        try:
            llm = get_llm()
        except Exception as exc:  # missing key or package
            print(f"LLM not available ({exc}). Continuing in deterministic mode.")
    mem = Memory()
    agent = TriageAgent(mem, llm)

    print(BAR + "\nTriageGuide - chat mode" + (" (LLM)" if llm else " (no LLM)") + "\n" + BAR)
    code = input("Patient code from an earlier visit (Enter if first time) > ").strip().upper()
    try:
        turn = agent.start(code or None)
    except KeyError:
        print("That code was not found, so I will start a new record.")
        turn = agent.start(None)
    print("\n" + turn.message)
    if args.trace:
        show_trace(turn)

    while not agent.done:
        text = input("\n> ").strip()
        if text.lower() in ("quit", "exit"):
            break
        t0 = time.time()
        turn = agent.handle(text)
        elapsed = time.time() - t0
        print("\n" + (BAR + "\n" if turn.done else "") + turn.message + ("\n" + BAR if turn.done else ""))
        if args.trace:
            show_trace(turn)
            llm_s = sum(t.get("total_s", 0) for t in turn.tool_log if t["tool"] == "llm_calls")
            print(f"   [time] {elapsed:.1f}s total = {llm_s:.1f}s in the LLM + {elapsed - llm_s:.1f}s everything else")
    if agent.done:
        lat = agent.latency_summary()
        if args.trace and lat["llm_calls"]:
            print(f"\nLatency summary: {lat['llm_calls']} LLM calls, {lat['llm_seconds']}s in total")
            for label, b in lat["by_purpose"].items():
                print(f"   {label:8s} calls={b['calls']}  total={b['seconds']}s  avg={b['seconds'] / b['calls']:.1f}s  "
                      f"max={b['max_s']}s  rate-limit hits={b['rate_limit_hits']} (waited {b['sleep_s']}s)  "
                      f"output tokens={b['completion_tokens']}  failed={b['failed']}")
        lines = mem.timeline_text(agent.state.patient_id)
        if lines:
            print("\nSymptom timeline (for your doctor):")
            for line in lines:
                print("  " + line)
    mem.close()


if __name__ == "__main__":
    main()
