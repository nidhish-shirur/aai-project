"""Ablations: switch ONE safety component off at a time and re-run the same cases.

Purpose: show what each part of the design actually contributes (and prove it is not decoration).
Each ablation is a context manager that temporarily replaces a function and restores it on exit.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from unittest import mock

import agent.orchestrator as orch
import engine.triage_rules as rules
from engine.red_flags import RedFlagResult

ABLATIONS = {
    "full_system": "Everything on (reference)",
    "no_red_flag_gate": "No emergency gate at all (neither words nor facts)",
    "no_text_gate": "Emergency gate reads extracted facts only, not the patient's words",
    "no_risk_modifier": "No age / chronic-condition / pregnancy risk modifier",
    "no_cautious_floor": "No 'cautious level when questions are unanswered' floor",
    "no_protocol_rules": "No protocol rules (only emergency gate, defaults and modifier)",
}


@contextmanager
def ablation(name: str):
    if name not in ABLATIONS:
        raise KeyError(name)
    with ExitStack() as stack:
        patch = stack.enter_context
        if name == "no_red_flag_gate":
            patch(mock.patch.object(orch, "check_text", lambda text: []))
            patch(mock.patch.object(orch, "check_red_flags", lambda state, protocol=None: RedFlagResult()))
            patch(mock.patch.object(rules, "check_red_flags", lambda state, protocol=None: RedFlagResult()))
        elif name == "no_text_gate":
            import engine.red_flags as rf
            patch(mock.patch.object(orch, "check_text", lambda text: []))
            patch(mock.patch.object(rf, "check_text", lambda text: []))
        elif name == "no_risk_modifier":
            patch(mock.patch.object(rules, "risk_modifier_reasons", lambda state: []))
        elif name == "no_cautious_floor":
            patch(mock.patch.object(rules, "required_unanswered", lambda state, protocol: []))
        elif name == "no_protocol_rules":
            patch(mock.patch.object(rules, "evaluate", lambda cond, facts: False))
        yield
