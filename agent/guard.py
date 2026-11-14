"""Output guard and patient-request detector (Phase 6).

Purpose: the product promise is "decision support, never diagnosis, never treatment".
The LLM writes only a few words the patient sees (the question wording and, later, the
summary), so every piece of LLM-written text passes through `check_output` first. If it
trips, the caller discards it and uses fixed wording from the JSON knowledge base.
`detect_requests` spots a patient ASKING for a diagnosis or medicine so the agent can
refuse in fixed words and carry on with triage.

Both are plain regex, deliberately conservative: a false alarm only costs a fallback to
the protocol's own wording, a miss costs trust.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CONDITIONS = (r"flu|influenza|covid\w*|corona\w*|dengue|malaria|typhoid|pneumonia|meningitis|appendicitis|"
              r"migraine|heart attack|stroke|tuberculosis|tb|gastroenteritis|food poisoning|uti|"
              r"urinary tract infection|viral|bacterial|cancer|tumou?r|diabetes|asthma attack|sepsis|"
              r"infection|allerg\w+|anxiety|depression")

DIAGNOSIS_OUT = re.compile(
    r"\b(diagnos\w*|"
    r"you (may|might|could|probably|likely|definitely|certainly|seem to) (have|be suffering|be having)|"
    r"you (have|are suffering from|are having|are diagnosed with) (a |an |the )?(" + CONDITIONS + r")\b|"
    r"sounds like|looks like (you|it|this)|seems like (you|it|this)|"
    r"(this|it) (is|could be|might be|may be|is probably|is likely) (a |an |the )?(" + CONDITIONS + r")\b|"
    r"likely (cause|condition|diagnosis)|suffering from|symptoms of (a |an )?(" + CONDITIONS + r"))\b", re.I)

TREATMENT_OUT = re.compile(
    r"\b(medicine\w*|medication\w*|tablet\w*|pill\w*|capsule\w*|syrup\w*|dose\w*|dosage|prescri\w*|"
    r"treatment\w*|therapy|antibiotic\w*|painkiller\w*|antihistamine\w*|steroid\w*|ointment\w*|inhaler\w*|"
    r"paracetamol|acetaminophen|ibuprofen|aspirin|crocin|dolo|"
    r"\d+(\.\d+)? ?(mg|ml|mcg)|"
    r"you should (take|use|apply|try|drink|eat|avoid)|"
    r"i (recommend|suggest|advise|prescribe) (you )?(take|use|try|that)|"
    r"home remed\w*|treat(ed|ing)? (it|this|with))\b", re.I)


@dataclass
class GuardResult:
    ok: bool
    categories: list[str] = field(default_factory=list)
    matches: list[str] = field(default_factory=list)


def check_output(text: str) -> GuardResult:
    """Is this LLM-written text free of diagnosis and treatment language?"""
    cats, found = [], []
    for name, rx in (("diagnosis", DIAGNOSIS_OUT), ("treatment", TREATMENT_OUT)):
        hits = [m.group(0) for m in rx.finditer(text or "")]
        if hits:
            cats.append(name)
            found += hits
    return GuardResult(not cats, cats, found)


def safe_or(text: str, fallback: str) -> tuple[str, bool]:
    """Return (text, False) if clean, else (fallback, True)."""
    return (text, False) if check_output(text).ok else (fallback, True)


# ---------------------------------------------------------------- what the PATIENT asks for
_COND = CONDITIONS
DIAGNOSIS_REQUEST = re.compile(
    r"\b(diagnose|diagnosis|"
    r"what('?s| is| are)? (wrong|the matter) with (me|him|her|my \w+)|"
    r"what (do|does|did) (i|he|she|my \w+) have|what have i got|"
    r"what (disease|illness|sickness|condition|infection|problem) (do|does|have|has|am|is)\b|"
    r"which (disease|illness|condition|infection)|"
    r"(do|does|did|could|can|might) (i|he|she|my \w+|it|this) (have|be) (a |an )?(" + _COND + r")\b|"
    r"(am i|is he|is she|is it|is this|is that) (a |an |having |suffering from )?(" + _COND + r")\b|"
    r"tell me (what|which) (disease|illness|condition) )", re.I)

TREATMENT_REQUEST = re.compile(
    r"\b((what|which)\b.{0,30}\b(medicine|medication|tablet|pill|drug|antibiotic|painkiller|syrup|treatment)\w*|"
    r"should (i|he|she) (take|use|buy|have|give)\b.{0,40}\b(medicine|medication|tablet|pill|paracetamol|"
    r"ibuprofen|aspirin|antibiotic|dolo|crocin|syrup)\w*|"
    r"how (much|many)\b.{0,40}\b(paracetamol|ibuprofen|aspirin|tablet|pill|mg|dose|medicine|crocin|dolo)\w*|"
    r"\b(dose|dosage)\b|prescribe|prescription|home remed\w*|"
    r"how (do|can|should) (i|we) (treat|cure)\b|cure for\b|treatment for\b)", re.I)


def detect_requests(text: str) -> list[str]:
    """Which forbidden things did the patient ask for? Subset of ['diagnosis', 'treatment']."""
    kinds = []
    if DIAGNOSIS_REQUEST.search(text or ""):
        kinds.append("diagnosis")
    if TREATMENT_REQUEST.search(text or ""):
        kinds.append("treatment")
    return kinds
