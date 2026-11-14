"""Prints the vignette evaluation table.   Run: python eval/run_vignettes.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.vignette_runner import metrics, run_all  # noqa: E402
from engine.protocols import URGENCY_LEVELS  # noqa: E402


def main() -> None:
    outcomes = run_all()
    print(f"{'ID':<5}{'Gold':<11}{'Predicted':<11}{'Conf':<8}Result")
    print("-" * 52)
    for o in outcomes:
        flag = "ok" if o.diff == 0 else ("OVER-triage" if o.diff > 0 else "UNDER-TRIAGE !!")
        print(f"{o.vignette['id']:<5}{o.gold:<11}{o.predicted:<11}{o.result.confidence:<8}{flag}")

    m = metrics(outcomes)
    print("\nConfusion matrix (rows = gold, columns = predicted)")
    print(f"{'':<11}" + "".join(f"{lv:<11}" for lv in URGENCY_LEVELS))
    for g in URGENCY_LEVELS:
        row = [sum(1 for o in outcomes if o.gold == g and o.predicted == p) for p in URGENCY_LEVELS]
        print(f"{g:<11}" + "".join(f"{c:<11}" for c in row))

    print(f"\nVignettes:                  {m['n']}")
    print(f"Emergency recall:           {m['emergency_recall']:.0%}   (target 100%)")
    print(f"Under-triage rate:          {m['under_triage']:.0%}   (target <= 5%)")
    print(f"Exact agreement:            {m['exact']:.0%}   (target >= 80%)")
    print(f"Within one level:           {m['within_one']:.0%}   (target >= 95%)")
    print(f"Over-triage rate:           {m['over_triage']:.0%}   (reported; safe direction)")


if __name__ == "__main__":
    main()
