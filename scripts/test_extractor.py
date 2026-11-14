"""Phase 2 live check: run the real LLM extractor on sample messages.

Run from the project root:   python scripts/test_extractor.py
Needs your GROQ_API_KEY in .env. Prints what the model extracted and what the
validator accepted or rejected. Paste the output back if anything looks wrong.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from agent.extractor import extract_into_state  # noqa: E402
from agent.llm_client import LLMError, get_llm  # noqa: E402
from agent.state import PatientState  # noqa: E402

CASES = [
    ("I am 42 years old male, fever for three days, temperature 101.3 F", None),
    ("mujhe kal se bahut tez sar dard hai aur ulti bhi ho rahi hai", None),
    ("My father has chest pain and he is sweating a lot", None),
    ("No chest pain and no difficulty breathing, just a mild cough for 10 days", None),
    ("not really", "stiff_neck"),
    ("I haven't measured it", "temperature_c"),
    ("two weeks", "duration_days"),
    ("I have diabetes and take sugar tablets", "chronic_conditions"),
]


def main() -> int:
    try:
        llm = get_llm()
    except LLMError as exc:
        print(exc)
        return 1
    print(f"Provider: {config.LLM_PROVIDER}   Model: {config.GROQ_MODEL}\n")
    failures = 0
    for text, pending in CASES:
        state = PatientState(complaint="fever" if pending else None)
        start = time.time()
        result = extract_into_state(state, text, llm, pending_key=pending)
        took = time.time() - start
        print(f"> {text!r}   (pending: {pending})")
        print(f"  complaint : {state.complaint}")
        print(f"  applied   : {result.applied}")
        if result.dont_know:
            print(f"  dont_know : {result.dont_know}")
        if result.rejected:
            print(f"  rejected  : {result.rejected}")
        if result.unverified_true:
            print(f"  unverified yes: {result.unverified_true}")
        if result.used_fallback:
            print(f"  !! LLM FAILED, fallback used: {result.error}")
            failures += 1
        print(f"  time      : {took:.1f}s\n")
    print("Done." if not failures else f"Done with {failures} LLM failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
