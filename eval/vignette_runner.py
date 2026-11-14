"""Runs the rule engine on the gold vignettes (no LLM needed).

Purpose: measure the safety core. A "simulated patient" answers every
required question from the vignette; anything the vignette does not mention
is answered 'no' (False / empty list), like a patient with nothing else to report.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import config
from agent.state import PatientState
from engine.conditions import evaluate
from engine.protocols import get_protocol, level_index, ordered_fields
from engine.triage_rules import TriageResult, compute_urgency


def load_vignettes(path=None) -> list[dict]:
    path = path or config.DATA_DIR / "gold_vignettes.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def simulate_state(vignette: dict) -> PatientState:
    state = PatientState(complaint=vignette["complaint"])
    for msg in vignette["messages"]:
        state.add_user_message(msg)
    state.update(vignette["facts"])
    protocol = get_protocol(vignette["complaint"])
    for f in ordered_fields(protocol):  # in asking order, so ask_if sees earlier answers
        if not f.required or state.get(f.key) is not None:
            continue
        if f.ask_if is not None and not evaluate(f.ask_if, state.all_facts()):
            continue
        if f.type == "bool":
            state.update({f.key: False})
        elif f.type == "list":
            state.update({f.key: []})
    return state


@dataclass
class Outcome:
    vignette: dict
    result: TriageResult

    @property
    def gold(self) -> str:
        return self.vignette["gold"]

    @property
    def predicted(self) -> str:
        return self.result.urgency

    @property
    def diff(self) -> int:
        return level_index(self.predicted) - level_index(self.gold)


def run_all(vignettes: list[dict] | None = None) -> list[Outcome]:
    vignettes = vignettes or load_vignettes()
    return [Outcome(v, compute_urgency(simulate_state(v))) for v in vignettes]


def metrics(outcomes: list[Outcome]) -> dict:
    n = len(outcomes)
    emergencies = [o for o in outcomes if o.gold == "emergency"]
    return {
        "n": n,
        "exact": sum(o.diff == 0 for o in outcomes) / n,
        "within_one": sum(abs(o.diff) <= 1 for o in outcomes) / n,
        "under_triage": sum(o.diff < 0 for o in outcomes) / n,
        "over_triage": sum(o.diff > 0 for o in outcomes) / n,
        "emergency_recall": (sum(o.predicted == "emergency" for o in emergencies) / len(emergencies))
        if emergencies else 1.0,
    }
