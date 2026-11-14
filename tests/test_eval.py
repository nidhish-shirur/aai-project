"""Tests for the Phase 8 evaluation code (so the numbers in the report can be trusted)."""
import csv
import hashlib
import json

import pytest

import config
from eval import extraction_eval, latency_report, metrics, run_eval
from eval.ablations import ABLATIONS, ablation
from eval.baselines import parse_level, transcript_text
from eval.sim_patient import answer_text, run_case


# ---------------- metrics
def test_summarize_counts_under_over_and_dangerous():
    pairs = [("emergency", "moderate"), ("urgent", "low"), ("low", "urgent"), ("moderate", "moderate")]
    s = metrics.summarize(pairs)
    assert s["n"] == 4 and s["exact"] == 0.25
    assert s["under_triage"] == 0.5 and s["over_triage"] == 0.25
    assert s["dangerous_under_triage"] == 2
    assert s["emergency_recall"] == 0.0


def test_emergency_to_urgent_is_under_triage_but_not_dangerous():
    s = metrics.summarize([("emergency", "urgent")])
    assert s["under_triage"] == 1.0 and s["dangerous_under_triage"] == 0 and s["emergency_recall"] == 0.0


def test_perfect_and_confusion_matrix_shape():
    pairs = [(l, l) for l in metrics.LEVELS]
    s = metrics.summarize(pairs)
    assert s["exact"] == 1.0 and s["weighted_kappa"] == pytest.approx(1.0)
    assert [row[i] for i, row in enumerate(s["confusion"])] == [1, 1, 1, 1]


def test_wilson_interval_sane():
    lo, hi = metrics.wilson(8, 10)
    assert 0.4 < lo < 0.8 < hi < 1.0
    assert metrics.wilson(0, 0) == (0.0, 0.0) or metrics.wilson(0, 0)[1] <= 1


def test_percentile_and_cohen():
    assert metrics.percentile([1, 2, 3, 4, 5], 50) == 3
    assert metrics.percentile([], 50) is None
    assert metrics.cohen_kappa_nominal(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)


def test_by_group():
    rows = [{"group": "g", "gold": "low", "pred": "low"}, {"group": "g", "gold": "low", "pred": "urgent"}]
    g = metrics.by_group(rows)
    assert g["g"]["n"] == 2 and g["g"]["exact"] == 0.5


# ---------------- simulated patient
def test_answer_text():
    assert answer_text("bool", "x", {"x": True}) == "yes"
    assert answer_text("bool", "x", {"x": False}) == "no"
    assert answer_text("bool", "x", {}) == "no"          # unmentioned symptom = not present
    assert answer_text("float", "t", {}) == "I don't know"
    assert answer_text("float", "t", {"t": 38.5}) == "38.5"
    assert answer_text("list", "c", {"c": []}) == "none"
    assert "heart disease" in answer_text("list", "c", {"c": ["heart_disease"]})


# ---------------- cases and freezing
def test_case_file_is_valid_and_frozen():
    cases = run_eval.load_cases()
    assert len(cases) == 48 and len({c["id"] for c in cases}) == 48
    assert all(c["gold"] in metrics.LEVELS for c in cases)
    assert run_eval.labels_frozen_ok(), "heldout_cases.json changed after freezing"


def test_checksum_detects_edit(tmp_path, monkeypatch):
    f = tmp_path / "c.json"
    f.write_text("[]")
    s = tmp_path / "c.sha256"
    s.write_text(hashlib.sha256(b"[]").hexdigest())
    monkeypatch.setattr(run_eval, "CASES", f)
    monkeypatch.setattr(run_eval, "CHECKSUM", s)
    assert run_eval.labels_frozen_ok()
    f.write_text("[1]")
    assert not run_eval.labels_frozen_ok()


def test_run_case_emergency_text_without_llm():
    case = next(c for c in run_eval.load_cases() if c["id"] == "H01")
    r = run_case(case, llm=None)
    assert r["done"] and r["pred"] == "emergency"


def test_every_case_finishes_offline():
    for c in run_eval.load_cases():
        r = run_case(c, llm=None)
        assert r["done"] and r["pred"] in metrics.LEVELS, c["id"]


# ---------------- ablations
def test_ablation_changes_then_restores():
    case = next(c for c in run_eval.load_cases() if c["id"] == "H01")
    before = run_case(case)["pred"]
    with ablation("no_red_flag_gate"):
        assert run_case(case)["pred"] != "emergency"
    assert run_case(case)["pred"] == before == "emergency"


@pytest.mark.parametrize("name", list(ABLATIONS))
def test_every_ablation_runs_and_restores(name):
    case = run_eval.load_cases()[0]
    ref = run_case(case)["pred"]
    with ablation(name):
        run_case(case)
    assert run_case(case)["pred"] == ref


def test_unknown_ablation():
    with pytest.raises(KeyError):
        with ablation("nope"):
            pass


# ---------------- baseline parsing
def test_parse_level():
    assert parse_level({"urgency": " Urgent "}) == "urgent"
    assert parse_level({"urgency": "critical"}) is None
    assert parse_level("urgent") is None
    assert parse_level({}) is None


def test_transcript_drops_greeting():
    t = transcript_text([{"role": "assistant", "content": "Hello"}, {"role": "user", "content": "fever"},
                         {"role": "assistant", "content": "How long?"}])
    assert "Hello" not in t and "Patient: fever" in t and "Assistant: How long?" in t


# ---------------- extraction scoring
def test_extraction_false_denial_and_extra():
    case = {"id": "T", "gold": {"fever": True, "cough": True}, "complaint": "fever", "dont_know": []}
    s = extraction_eval.score_case(case, {"fever": False, "rash": True}, [], "cough_breathing")
    assert s["false_denial"] == ["fever"] and s["missed_true"] == ["cough"] and s["extra_true"] == ["rash"]
    assert s["complaint_ok"] is False


def test_extraction_perfect():
    case = {"id": "T", "gold": {"fever": True}, "complaint": "fever", "dont_know": ["x"]}
    s = extraction_eval.score_case(case, {"fever": True}, ["x"], "fever")
    assert (s["tp"], s["fp"], s["fn"]) == (1, 0, 0) and s["dont_know_ok"] and s["complaint_ok"]


def test_extraction_cases_file():
    cases = extraction_eval.load_cases()
    assert len(cases) == 30 and len({c["id"] for c in cases}) == 30


# ---------------- latency collect
def test_latency_collect():
    rep = {"tool_log": [{"tool": "extract_facts", "fast_path": True, "turn": 1},
                        {"tool": "llm_calls", "turn": 1, "total_s": 3.0,
                         "calls": [{"label": "plan", "seconds": 1.0, "ok": True},
                                   {"label": "extract", "seconds": 2.0, "ok": False, "error": "Request timed out",
                                    "rate_limit_hits": 1}]}]}
    c = latency_report.collect([rep])
    assert c["fast_path_answers"] == 1 and c["timeouts"] == 1 and c["rate_limit_hits"] == 1
    assert c["by_purpose"]["plan"]["n"] == 1


# ---------------- clinician sheet import
def test_load_reviewed(tmp_path):
    p = tmp_path / "r.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "text", "facts", "clinician_label"])
        w.writerow(["H01", "t", "f", "Emergency"])
        w.writerow(["H02", "t", "f", ""])
        w.writerow(["H03", "t", "f", "bogus"])
    assert run_eval.load_reviewed(str(p)) == {"H01": "emergency"}


def test_review_sheet_hides_ai_labels():
    with open(config.EVAL_DIR / "review_sheet.csv", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    header = next(r for r in rows if r and r[0] == "id")
    assert [h.lower() for h in header] == ["id", "what the patient says", "additional facts", "clinician_label", "comment"]
    body = [r for r in rows[rows.index(header) + 1:] if r]
    assert len(body) == 48 and all(r[3] == "" for r in body)       # label column empty: AI labels not leaked
    assert not any("rationale" in " ".join(r).lower() for r in body)


# ---------------- routing fixes found by the Phase 8 failure analysis
from agent.extractor import safer_complaint
from engine.protocols import _trigger_regex, select_protocol_from_text as route


def test_negated_keyword_does_not_route():
    assert route("I have no chest pain and no breathing problems, just a mild headache") == "headache"
    assert route("no fever, no back pain, burning when I pee") == "urinary"
    assert route("I have chest pain") == "chest_pain"
    assert route("no fever but chest pain with sweating") == "chest_pain"   # 'but' cancels the negation


def test_burning_is_not_a_burn_injury():
    assert route("burning when I pee for 2 days") == "urinary"
    assert route("I burnt my hand on the stove") == "minor_injury"
    assert route("I got a burn on my arm") == "minor_injury"


def test_trigger_regex_whole_word_marker():
    import re
    assert re.search(_trigger_regex("burn$"), "a burn")
    assert not re.search(_trigger_regex("burn$"), "burning")
    assert re.search(_trigger_regex("cough"), "coughing")      # ordinary triggers still match word starts


def test_safer_complaint_ignores_negated_keyword():
    # a correct LLM pick (headache) must not be overridden by a negated chest-pain mention
    assert safer_complaint("headache", "no chest pain, just a headache") == "headache"
    # but a real higher-priority symptom still wins over a lower LLM pick
    assert safer_complaint("headache", "I have a headache and chest pain") == "chest_pain"
