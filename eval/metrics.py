"""Evaluation metrics (pure functions, no I/O).

Triage is an ORDERED 4-level problem (low < moderate < urgent < emergency), and the two mistakes are
not equal: sending an emergency home (under-triage) is dangerous, sending a cold to the clinic
(over-triage) is only wasteful. So besides accuracy we report both directions separately.
"""
from __future__ import annotations

from math import sqrt
from typing import Iterable, Optional

LEVELS = ["low", "moderate", "urgent", "emergency"]
IDX = {lv: i for i, lv in enumerate(LEVELS)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval for a proportion k/n (honest about small samples)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def confusion(pairs: Iterable[tuple[str, str]]) -> list[list[int]]:
    """Rows = gold, columns = predicted, in LEVELS order."""
    m = [[0] * 4 for _ in range(4)]
    for gold, pred in pairs:
        m[IDX[gold]][IDX[pred]] += 1
    return m


def weighted_kappa(pairs: list[tuple[str, str]]) -> Optional[float]:
    """Linear-weighted Cohen's kappa: agreement beyond chance, counting near-misses as partial agreement."""
    n = len(pairs)
    if n == 0:
        return None
    m = confusion(pairs)
    row = [sum(r) for r in m]
    col = [sum(m[i][j] for i in range(4)) for j in range(4)]
    obs = exp = 0.0
    for i in range(4):
        for j in range(4):
            w = abs(i - j) / 3
            obs += w * m[i][j] / n
            exp += w * row[i] * col[j] / (n * n)
    return None if exp == 0 else 1 - obs / exp


def summarize(pairs: list[tuple[str, str]]) -> dict:
    """pairs = [(gold, predicted)]. Safety-relevant numbers first."""
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    d = [IDX[p] - IDX[g] for g, p in pairs]
    emerg = [(g, p) for g, p in pairs if g == "emergency"]
    serious = [(g, p) for g, p in pairs if g in ("urgent", "emergency")]
    emerg_hit = sum(p == "emergency" for _, p in emerg)
    serious_hit = sum(IDX[p] >= IDX["urgent"] for _, p in serious)
    # dangerous = an emergency sent to 'moderate' or 'low', or an urgent case sent to 'low'
    dangerous = sum((g == "emergency" and IDX[p] <= IDX["moderate"]) or (g == "urgent" and p == "low")
                    for g, p in pairs)
    f1s = []
    for lv in LEVELS:
        tp = sum(1 for g, p in pairs if g == lv and p == lv)
        fp = sum(1 for g, p in pairs if g != lv and p == lv)
        fn = sum(1 for g, p in pairs if g == lv and p != lv)
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return {
        "n": n,
        "exact": sum(x == 0 for x in d) / n,
        "exact_ci": wilson(sum(x == 0 for x in d), n),
        "within_one": sum(abs(x) <= 1 for x in d) / n,
        "under_triage": sum(x < 0 for x in d) / n,
        "over_triage": sum(x > 0 for x in d) / n,
        "emergency_recall": emerg_hit / len(emerg) if emerg else None,
        "emergency_recall_ci": wilson(emerg_hit, len(emerg)) if emerg else None,
        "urgent_or_higher_recall": serious_hit / len(serious) if serious else None,
        "dangerous_under_triage": dangerous,
        "weighted_kappa": weighted_kappa(pairs),
        "macro_f1": sum(f1s) / len(f1s) if f1s else None,
        "confusion": confusion(pairs),
    }


def by_group(rows: list[dict], key: str = "group") -> dict[str, dict]:
    """rows = [{'gold','pred','group',...}] -> summary per group."""
    groups: dict[str, list[tuple[str, str]]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append((r["gold"], r["pred"]))
    return {g: summarize(p) for g, p in sorted(groups.items())}


def percentile(values: list[float], q: float) -> Optional[float]:
    """q in [0,100], linear interpolation (same as numpy's default)."""
    if not values:
        return None
    v = sorted(values)
    if len(v) == 1:
        return v[0]
    pos = (len(v) - 1) * q / 100
    lo, hi = int(pos), min(int(pos) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def cohen_kappa_nominal(a: list[str], b: list[str]) -> Optional[float]:
    """Plain Cohen's kappa between two raters (for AI-draft vs clinician labels)."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return None if pe == 1 else (po - pe) / (1 - pe)
