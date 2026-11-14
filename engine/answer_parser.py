"""Turns a typed answer into a typed value for a protocol field.

Purpose: the deterministic fallback mode (no LLM) needs to understand short
answers such as 'yes', '38.5', '3 days', 'none'. Returns (status, value) where
status is 'ok', 'unknown' (patient does not know) or 'invalid'.
"""
from __future__ import annotations

import re
from typing import Any

from engine.protocols import Field_

YES = {"y", "yes", "yeah", "yep", "yup", "true", "haan", "ha", "han", "ji", "sure", "correct"}
NO = {"n", "no", "nope", "nah", "false", "nahi", "nahin", "na", "none"}
UNKNOWN_PHRASES = ("dk", "idk", "don't know", "dont know", "do not know", "not sure", "unsure",
                   "no idea", "unknown", "pata nahi", "not measured", "haven't measured", "have not measured")
WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "half": 0.5, "a": 1, "an": 1}
CONDITION_WORDS = {
    "diabet": "diabetes", "sugar": "diabetes", "heart": "heart_disease", "cardiac": "heart_disease",
    "asthma": "lung_disease", "copd": "lung_disease", "lung": "lung_disease",
    "kidney": "kidney_disease", "renal": "kidney_disease", "liver": "liver_disease",
    "cancer": "cancer", "chemo": "cancer", "hiv": "immunocompromised", "steroid": "immunocompromised",
    "immun": "immunocompromised", "transplant": "immunocompromised",
    "hypertension": "hypertension", "blood pressure": "hypertension", "bp": "hypertension",
}


def _number(text: str):
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if m:
        return float(m.group())
    for word in re.findall(r"[a-z]+", text):
        if word in WORD_NUMBERS and word not in ("a", "an"):
            return float(WORD_NUMBERS[word])
    if re.search(r"\b(a|an)\b", text):
        return 1.0
    return None


def parse_answer(field: Field_, raw: str) -> tuple[str, Any]:
    text = raw.strip().lower().replace("’", "'")
    if not text:
        return "invalid", None
    if any(p in text for p in UNKNOWN_PHRASES):
        return "unknown", None

    if field.type == "bool":
        words = set(re.findall(r"[a-z']+", text))
        if words & YES and not words & NO:
            return "ok", True
        if words & NO and not words & YES:
            return "ok", False
        return "invalid", None

    if field.type in ("float", "int"):
        num = _number(text)
        if num is None:
            return "invalid", None
        if field.key == "temperature_c" and (num >= 45 or "f" in re.findall(r"[a-z]+", text)):
            num = round((num - 32) * 5 / 9, 1)  # Fahrenheit -> Celsius
        if field.key == "age_years":
            if "month" in text:
                num = num / 12
            elif "week" in text:
                num = num / 52
        if field.key == "duration_days":
            if "week" in text:
                num *= 7
            elif "month" in text:
                num *= 30
            elif "hour" in text:
                num /= 24
            elif "today" in text:
                num = 0.5
        if field.type == "int":
            num = int(round(num))
        return "ok", num

    if field.type == "enum":
        options = field.options or []
        if text.isdigit() and 1 <= int(text) <= len(options):
            return "ok", options[int(text) - 1]
        for opt in options:
            if text == opt or text == opt.replace("_", " "):
                return "ok", opt
        for opt in options:
            if opt.replace("_", " ") in text or opt in text:
                return "ok", opt
        if field.key == "pain_location":
            for opt, hints in {"lower_right": ["right"], "lower_left": ["left"],
                               "central": ["navel", "middle", "button"], "upper": ["top"],
                               "all_over": ["everywhere", "whole"]}.items():
                if any(h in text for h in hints):
                    return "ok", opt
        if field.key == "sex":
            if text in ("m", "man", "boy"):
                return "ok", "male"
            if text in ("f", "woman", "girl"):
                return "ok", "female"
        return "invalid", None

    if field.type == "list":
        words = set(re.findall(r"[a-z']+", text))
        if words & NO and not any(k in text for k in CONDITION_WORDS):
            return "ok", []
        found = []
        for key, canonical in CONDITION_WORDS.items():
            if key in text and canonical not in found:
                found.append(canonical)
        return "ok", found or ["other"]

    return "ok", raw.strip()
