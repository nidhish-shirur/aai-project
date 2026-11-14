"""Phase 1 test: validate every knowledge file.

Run from the project root:   python scripts/check_knowledge.py
Exit code 0 = all good. It catches typos in rules (for example a rule that
mentions a fact that no question ever collects, which would silently never fire).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.conditions import referenced_facts, validate_condition  # noqa: E402
from engine.protocols import (  # noqa: E402
    URGENCY_LEVELS, get_common_fields, get_protocols, get_red_flag_config, known_fact_keys,
)


def check() -> list[str]:
    errors: list[str] = []
    protocols = get_protocols()
    if "other" not in protocols:
        errors.append("Missing the fallback protocol 'other'")

    common_keys = {f.key for f in get_common_fields()}
    cfg = get_red_flag_config()
    extra_keys = {f["key"] for f in cfg["extra_facts"]}

    # global red flags
    flag_ids = set()
    for flag in cfg["flags"]:
        if flag["id"] in flag_ids:
            errors.append(f"red_flags.json: duplicate id {flag['id']}")
        flag_ids.add(flag["id"])
        if not (flag.get("patterns") or flag.get("patterns_all") or flag.get("when")):
            errors.append(f"red_flags.json: {flag['id']} has no pattern or condition")
        if "when" in flag:
            try:
                validate_condition(flag["when"])
                for fact in referenced_facts(flag["when"]) - extra_keys - common_keys:
                    errors.append(f"red_flags.json: {flag['id']} uses undeclared fact '{fact}'")
            except ValueError as exc:
                errors.append(f"red_flags.json: {flag['id']}: {exc}")

    for pid, proto in protocols.items():
        where = f"protocols/{pid}"
        keys = known_fact_keys(proto)
        field_keys = [f.key for f in proto.fields]
        if len(field_keys) != len(set(field_keys)):
            errors.append(f"{where}: duplicate field keys")
        for f in proto.fields:
            if f.type == "enum" and not f.options:
                errors.append(f"{where}: enum field {f.key} has no options")
            if f.ask_if is not None:
                try:
                    validate_condition(f.ask_if)
                    for fact in referenced_facts(f.ask_if) - keys:
                        errors.append(f"{where}: ask_if of {f.key} uses unknown fact '{fact}'")
                except ValueError as exc:
                    errors.append(f"{where}: ask_if of {f.key}: {exc}")
        ids = [r.id for r in proto.rules] + [r.id for r in proto.red_flags]
        if len(ids) != len(set(ids)):
            errors.append(f"{where}: duplicate rule or red-flag ids")
        for item in [*proto.rules, *proto.red_flags]:
            try:
                validate_condition(item.when)
                for fact in referenced_facts(item.when) - keys:
                    errors.append(f"{where}: '{item.id}' uses unknown fact '{fact}'")
            except ValueError as exc:
                errors.append(f"{where}: '{item.id}': {exc}")
        for r in proto.rules:
            if r.urgency == "emergency" and not r.reason:
                errors.append(f"{where}: emergency rule {r.id} has no reason")
        if proto.default_urgency not in URGENCY_LEVELS:
            errors.append(f"{where}: bad default_urgency")
        if not proto.warning_signs:
            errors.append(f"{where}: no warning_signs (needed for user advice)")
    return errors


if __name__ == "__main__":
    problems = check()
    protos = get_protocols()
    n_rules = sum(len(p.rules) + len(p.red_flags) for p in protos.values())
    print(f"Checked {len(protos)} protocols, {n_rules} protocol rules, "
          f"{len(get_red_flag_config()['flags'])} global red flags.")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(" -", p)
        raise SystemExit(1)
    print("Knowledge base OK.")
