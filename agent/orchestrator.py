"""The triage agent (orchestrator): one patient message in, one agent reply out.

Purpose: this is the "agentic" part. Each turn the agent chooses an ACTION with
tools, but the LLM never has the last word on safety:

    patient text -> [1 red_flag_gate]   words only, no LLM   -> emergency? stop here
                 -> [2 extract_facts]   LLM proposes, Python validates
                 -> [3 red_flag_gate]   again, now with the extracted facts
                 -> [4 plan_next]       LLM picks: ask_question / compute_urgency / escalate_to_clinician
                                        (Python validates the plan and can override it)
                 -> [5 ask_question | compute_urgency | escalate_to_clinician]

Safety rules enforced in code, not in prompts:
* The emergency gate runs BEFORE any LLM call, so an emergency needs zero LLM calls.
* The LLM may only choose among fields the protocol still needs (no invented questions
  as data fields), cannot stop early while required fields are missing, and its question
  wording is rejected if it contains diagnosis or medicine language.
* Urgency comes from engine.triage_rules only. The LLM can ask for a human review
  (escalate), never lower anything.
* If the LLM fails at any step the agent falls back to the deterministic order used by
  scripts/cli_fallback.py, so the conversation always continues.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import config
from agent.extractor import extract_into_state
from agent.guard import check_output, detect_requests
from agent.llm_client import LLM, LLMError, TimedLLM
from agent.safety_review import REVIEW_SYSTEM, review_and_apply
from agent.state import PatientState
from engine import pathways
from engine.answer_parser import parse_answer
from engine.protocols import Field_, get_protocol, missing_fields, select_protocol_from_text
from engine.red_flags import check_red_flags, check_text
from engine.triage_rules import compute_urgency
from memory.db import Memory

MAX_INPUT_CHARS = 2000
MAX_QUESTION_CHARS = 240
CANDIDATES_SHOWN = 6
# Combined questions ("breathing? also chest pain?") caused wasted turns in the first live run:
# a bare "no" only answered the first field. Off by default; the plan validator ignores "also".
COMBINE_QUESTIONS = False
ACTIONS = ("ask_question", "compute_urgency", "escalate_to_clinician")

PLANNER_SYSTEM = """You are the planning step of a healthcare TRIAGE assistant. You never diagnose and never
suggest medicines or treatment. You only decide what to do next.

You receive: what is already known, the questions that are still open (candidates), and how many
questions are left. Reply with ONE JSON object:
{"action": "ask_question" | "compute_urgency" | "escalate_to_clinician",
 "field": "<key of ONE candidate>",
 "question": "<one short, plain, friendly question for the patient>",
 "reason": "<only for escalate_to_clinician: why a human should look>"}

Rules:
- Prefer candidates marked required. Prefer safety-relevant questions first.
- Never choose compute_urgency while required candidates remain.
- Ask about ONE field only. Never ask about something already known. No medical terms.
- The question must not contain a diagnosis, a guess about the cause, or advice about medicines.
- escalate_to_clinician only if the situation looks unusual or confusing and a human should review."""


@dataclass
class AgentTurn:
    kind: str                       # greeting | question | result | emergency | info | closed
    message: str
    done: bool = False
    urgency: Optional[str] = None
    result: Optional[dict] = None   # TriageResult.to_dict() when finished
    tool_log: list[dict] = field(default_factory=list)  # tools used during THIS turn
    refusal: str = ""              # fixed refusal text that was put in front of `message` (if any)


def _label(system: str) -> str:
    return "plan" if system == PLANNER_SYSTEM else "review" if system == REVIEW_SYSTEM else "extract"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TriageAgent:
    def __init__(self, memory: Memory, llm: Optional[LLM] = None,
                 clock: Callable[[], datetime] = _utcnow, max_questions: Optional[int] = None,
                 safety_review: bool = True):
        self.mem = memory
        if llm is not None and not isinstance(llm, TimedLLM):
            llm = TimedLLM(llm, _label)    # records time, retries and rate-limit waits of every call
        self.llm = llm                     # None = deterministic fallback mode
        self._llm_calls: list[dict] = []   # every LLM call of the session, with timings
        self.safety_review = safety_review
        self._refusal = ""                 # fixed refusal text to show with the next reply
        self.clock = clock
        self.max_questions = config.MAX_QUESTIONS if max_questions is None else max_questions
        self.state: Optional[PatientState] = None
        self.done = False
        self.tool_log: list[dict] = []     # whole session
        self.turn = 0
        self._turn_log: list[dict] = []
        self._pending: Optional[dict] = None      # what the last question was about
        self._misses: dict[str, int] = {}
        self._complaint_tries = 0
        self._escalation_reason: Optional[str] = None
        self._llm_failures = 0
        self._breaker_logged = False

    # ================================================================ public API
    @property
    def pending_kind(self) -> Optional[str]:
        """What the last agent message asked for: 'field', 'complaint', 'resolution' or None."""
        return self._pending["kind"] if self._pending else None

    @property
    def pending_key(self) -> Optional[str]:
        """The field key the last question was about (None if it was not a field question)."""
        keys = (self._pending or {}).get("keys") or []
        return keys[0] if keys else None

    def start(self, patient_code: Optional[str] = None) -> AgentTurn:
        """Open a session. Raises KeyError for an unknown patient code."""
        self.state = self.mem.start_session(patient_code, now=self.clock())
        s = self.state
        lines = [pathways.disclaimer(), f"If this is an emergency, call {config.EMERGENCY_NUMBER} now."]
        note = self.mem.continuity_note(s.patient_id, exclude_session_id=s.session_id, now=self.clock())
        if note:
            lines.append(note)
        earlier = self.mem.unresolved_earlier_symptoms(s.patient_id, s.session_id, now=self.clock())
        if earlier:
            self._pending = {"kind": "resolution", "keys": []}
            lines.append("Last time you told me about: " + ", ".join(e.lower() for e in earlier)
                         + ". Have those completely gone away? (yes / no)")
        else:
            self._pending = {"kind": "complaint", "keys": []}
            lines.append("What is the main problem today?")
        message = "\n\n".join(lines)
        self.mem.log_message(s.session_id, "agent", message, now=self.clock())
        self._log("start", patient_id=s.patient_id, earlier_unresolved=earlier)
        return AgentTurn("greeting", message, tool_log=self._flush())

    def handle(self, text: str) -> AgentTurn:
        """Process one patient message and return the agent's reply."""
        if self.state is None:
            raise RuntimeError("call start() first")
        if self.done:
            return AgentTurn("closed", "This assessment is complete. Please start a new session to begin again.",
                             done=True)
        text = (text or "").strip()[:MAX_INPUT_CHARS]
        if not text:
            return AgentTurn("info", "I did not catch that. Could you type your answer again?")
        self.turn += 1
        s = self.state
        self.mem.log_message(s.session_id, "user", text, now=self.clock())   # stored before anything can fail

        # 0. returning patient: "have your earlier symptoms gone away?"
        if self._pending and self._pending["kind"] == "resolution":
            self._pending = {"kind": "complaint", "keys": []}
            status, value = parse_answer(Field_(key="resolved", type="bool", question=""), text)
            if status == "ok":
                if value:
                    n = self.mem.mark_earlier_resolved(s.patient_id, s.session_id, now=self.clock())
                    self._log("mark_earlier_resolved", records=n)
                else:
                    self._log("earlier_symptoms_still_present")
                reply = "Thank you. What is the main problem today?"
                return self._reply(AgentTurn("question", reply, tool_log=self._flush()))
            # not a yes/no: treat it as the patient describing today's problem
            self._log("resolution_answer_unclear_treated_as_problem")

        # 1. deterministic emergency gate on the words alone (no LLM involved)
        hits = check_text(s.text_blob() + " . " + text)
        self._log("red_flag_gate", stage="text_only", triggered=bool(hits), labels=[h.label for h in hits])
        if hits:
            s.add_user_message(text)
            if s.complaint is None:
                s.complaint = select_protocol_from_text(text)
            return self._finish(emergency=True)

        # 1b. patient asks for a diagnosis or medicine: refuse in FIXED words, keep triaging
        asked = detect_requests(text)
        if asked:
            self._refusal = pathways.refusal(asked)
            self._log("refusal", kinds=asked)

        # 2. extraction (LLM proposes, Python validates; fallback if the LLM fails)
        keys = (self._pending or {}).get("keys") or []
        pending_key = keys[0] if keys else None
        llm_now = self._llm_for_call()
        ex = extract_into_state(s, text, llm_now, pending_key=pending_key)
        self._log("extract_facts", applied=sorted(ex.applied), rejected=[k for k, _ in ex.rejected],
                  dont_know=ex.dont_know, complaint=s.complaint, used_fallback=ex.used_fallback, fast_path=ex.fast_path,
                  error=ex.error or None)
        if ex.error:
            self._llm_failures += 1
        elif llm_now is not None and not ex.fast_path:
            self._llm_failures = 0      # only a REAL working call closes the breaker again
        if not (asked and not ex.applied and not ex.dont_know):   # a refusal-only message is not a missed answer
            self._track_misses(pending_key)

        # 3. gate again with facts and protocol-specific flags
        protocol = get_protocol(s.complaint) if s.complaint else None
        gate = check_red_flags(s, protocol)
        self._log("red_flag_gate", stage="text_and_facts", triggered=gate.triggered, labels=gate.labels)
        if gate.triggered:
            return self._finish(emergency=True)

        # 4. complaint still unknown? ask once more, then fall back to the generic protocol
        if s.complaint is None:
            self._complaint_tries += 1
            if self._complaint_tries < 2:
                self._pending = {"kind": "complaint", "keys": []}
                return self._reply(AgentTurn(
                    "question", "Could you tell me in a few words what the main problem is? "
                                "For example, fever, cough, pain, an injury or dizziness.",
                    tool_log=self._flush()))
            s.complaint = "other"
            self._log("complaint_defaulted_to_other")

        # 5. plan the next step and act on it
        plan = self._plan(s)
        if plan["action"] == "escalate_to_clinician":
            self._escalation_reason = plan.get("reason") or "Agent asked for human review"
            self._log("escalate_to_clinician", reason=self._escalation_reason)
            return self._finish(emergency=False)
        if plan["action"] == "compute_urgency":
            return self._finish(emergency=False)
        return self._ask(plan)

    # ================================================================ planning
    def _candidates(self, s: PatientState) -> list[Field_]:
        protocol = get_protocol(s.complaint)
        required = missing_fields(s, protocol)
        return required or missing_fields(s, protocol, include_optional=True)

    def _plan(self, s: PatientState) -> dict:
        """Decide the next action. Always returns a VALID plan:
        {"action", "field": Field_|None, "also": [Field_], "question": str, "source": "llm"|"rules"|"override"}"""
        protocol = get_protocol(s.complaint)
        candidates = self._candidates(s)
        required_left = bool(missing_fields(s, protocol))

        if s.questions_asked >= self.max_questions:
            self._log("plan_next", source="rules", action="compute_urgency", why="question budget used")
            return {"action": "compute_urgency"}
        if not candidates:
            self._log("plan_next", source="rules", action="compute_urgency", why="nothing left to ask")
            return {"action": "compute_urgency"}

        default = candidates[0]
        rules_plan = {"action": "ask_question", "field": default, "also": [], "question": default.question,
                      "source": "rules"}
        if self._llm_for_call() is None or len(candidates) == 1:
            self._log("plan_next", source="rules", action="ask_question", field=default.key,
                      why="no LLM" if self.llm is None else ("LLM failing" if self._llm_for_call() is None
                                                             else "only one question left"))
            return rules_plan

        try:
            raw = self.llm.complete_json(PLANNER_SYSTEM, self._planner_prompt(s, candidates))
        except LLMError as exc:
            self._llm_failures += 1
            self._log("plan_next", source="rules", action="ask_question", field=default.key,
                      why=f"planner failed: {exc}")
            return rules_plan
        self._llm_failures = 0              # the planner call worked: close the breaker
        plan, notes = self._validate_plan(raw, candidates, required_left)
        self._log("plan_next", source=plan["source"], action=plan["action"],
                  field=plan["field"].key if plan.get("field") else None, notes=notes)
        return plan

    def _planner_prompt(self, s: PatientState, candidates: list[Field_]) -> str:
        known = {k: v for k, v in s.all_facts().items() if v is not None}
        lines = [f"Complaint: {s.complaint}", f"Known: {known}", f"Patient could not answer: {s.unknown_fields}",
                 f"Questions asked so far: {s.questions_asked} of at most {self.max_questions}",
                 f"Last patient message: {s.user_messages[-1] if s.user_messages else ''}",
                 "Candidates (open questions, safest first):"]
        for f in candidates[:CANDIDATES_SHOWN]:
            lines.append(f'- key={f.key} required={f.required} type={f.type} default_question="{f.question}"')
        return "\n".join(lines)

    def _validate_plan(self, raw: Any, candidates: list[Field_], required_left: bool) -> tuple[dict, list[str]]:
        """Never trust the LLM's plan: check every part, fix or override what is wrong."""
        notes: list[str] = []
        by_key = {f.key: f for f in candidates}
        default = candidates[0]
        fallback = {"action": "ask_question", "field": default, "also": [], "question": default.question,
                    "source": "rules"}
        if not isinstance(raw, dict):
            return fallback, ["plan was not an object"]
        action = raw.get("action")
        if action not in ACTIONS:
            return fallback, [f"unknown action {action!r}"]

        if action == "escalate_to_clinician":
            reason = str(raw.get("reason") or "")[:200]
            return {"action": action, "reason": reason, "source": "llm"}, notes
        if action == "compute_urgency":
            if required_left:
                notes.append("override: required fields still missing")
                return {**fallback, "source": "override"}, notes
            return {"action": action, "source": "llm"}, notes

        field_obj = by_key.get(raw.get("field"))
        source = "llm"
        if field_obj is None:
            notes.append(f"override: field {raw.get('field')!r} is not an open candidate")
            field_obj, source = default, "override"
        if required_left and not field_obj.required:
            notes.append("override: an optional field was chosen while required ones remain")
            field_obj, source = default, "override"
        also = []
        extra = raw.get("also")
        if COMBINE_QUESTIONS and isinstance(extra, list) and extra and isinstance(extra[0], str):
            cand = by_key.get(extra[0])
            if cand and cand.key != field_obj.key and (cand.required or not required_left):
                also = [cand]
        question = str(raw.get("question") or "").strip()
        if source == "override" or not question:
            question = field_obj.question
            if not raw.get("question") and source == "llm":
                notes.append("empty question: used the protocol wording")
        elif not check_output(question).ok:
            notes.append("rejected question wording (diagnosis or medicine language): used protocol wording")
            question, source = field_obj.question, "override"
        elif len(question) > MAX_QUESTION_CHARS:
            notes.append("question too long: used protocol wording")
            question, source = field_obj.question, "override"
        if source == "override":
            also = []
        return {"action": "ask_question", "field": field_obj, "also": also, "question": question,
                "source": source}, notes

    # ================================================================ acting
    def _ask(self, plan: dict) -> AgentTurn:
        s = self.state
        f: Field_ = plan["field"]
        question = plan["question"]
        if self._misses.get(f.key):
            question = "Sorry, I did not quite catch that. " + question
        hint = f" ({'/'.join(f.options)})" if f.options and f.type == "enum" and f.key not in question else ""
        s.questions_asked += 1
        self._pending = {"kind": "field", "keys": [f.key] + [a.key for a in plan.get("also", [])]}
        self._log("ask_question", field=f.key, also=[a.key for a in plan.get("also", [])],
                  source=plan["source"], number=s.questions_asked)
        return self._reply(AgentTurn("question", question + hint, tool_log=self._flush()))

    def _track_misses(self, pending_key: Optional[str]) -> None:
        """If the patient did not answer the question just asked, retry once, then record 'unknown'."""
        s = self.state
        if not pending_key or not self._pending or self._pending.get("kind") != "field":
            return
        if s.get(pending_key) is None and pending_key not in s.unknown_fields:
            self._misses[pending_key] = self._misses.get(pending_key, 0) + 1
            if self._misses[pending_key] >= 2:
                s.mark_unknown(pending_key)
                self._log("gave_up_on_field", field=pending_key)
        else:
            self._misses.pop(pending_key, None)

    def _finish(self, emergency: bool) -> AgentTurn:
        s = self.state
        result = compute_urgency(s)
        if self._escalation_reason:
            result.needs_clinician_review = True
            if self._escalation_reason not in result.clinician_reasons:
                result.clinician_reasons.append("Flagged by the assistant: " + self._escalation_reason)
        rule_urgency = result.urgency
        review = None
        if self.safety_review and result.urgency != "emergency":
            review = review_and_apply(self._llm_for_call(), s, result)
            self._log("safety_review", **review.to_dict())
        self._log("compute_urgency", urgency=result.urgency, protocol=result.protocol_id,
                  confidence=result.confidence, red_flags=[h.label for h in result.red_flags],
                  matched_rules=result.matched_rules)
        report = result.to_dict()
        report["tool_log"] = list(self.tool_log) + list(self._turn_log)
        report["rule_urgency"] = rule_urgency
        report["llm_review"] = review.to_dict() if review else None
        self._collect_llm_timing()
        report["latency"] = self.latency_summary()
        report["questions_asked"] = s.questions_asked
        report["llm_used"] = self.llm is not None
        self.mem.save_state(s, now=self.clock())
        self.mem.record_symptom_events(s, now=self.clock())
        self.mem.save_report(s.session_id, report, now=self.clock())
        self.done = True
        message = render_result(result, s)
        refusal = ""
        if self._refusal and result.urgency != "emergency":
            refusal = self._refusal
            message = refusal + "\n\n" + message
        self._refusal = ""
        kind = "emergency" if result.urgency == "emergency" else "result"
        self.mem.log_message(s.session_id, "agent", message, now=self.clock())
        return AgentTurn(kind, message, done=True, urgency=result.urgency, result=report, tool_log=self._flush(),
                         refusal=refusal)

    # ================================================================ helpers
    def _llm_for_call(self) -> Optional[LLM]:
        """The LLM, unless it has failed 3 times in a row: then deterministic mode until a call works again
        (a dead connection would otherwise cost seconds on every turn)."""
        if self.llm is None:
            return None
        if self._llm_failures >= 3:
            if not self._breaker_logged:
                self._breaker_logged = True
                self._log("llm_circuit_open", failures=self._llm_failures)
            return None
        self._breaker_logged = False
        return self.llm

    def _reply(self, turn: AgentTurn) -> AgentTurn:
        s = self.state
        if self._refusal:
            turn.refusal = self._refusal
            turn.message = self._refusal + "\n\n" + turn.message
            self._refusal = ""
        self.mem.log_message(s.session_id, "agent", turn.message, now=self.clock())
        self.mem.save_state(s, now=self.clock())
        return turn

    def _log(self, tool: str, **detail: Any) -> None:
        self._turn_log.append({"turn": self.turn, "tool": tool, **detail})

    def _collect_llm_timing(self) -> None:
        """Move the timings recorded since the last turn into the tool log (one 'llm_calls' entry)."""
        if isinstance(self.llm, TimedLLM):
            calls = self.llm.drain()
            if calls:
                self._llm_calls.extend(calls)
                self._log("llm_calls", calls=calls, total_s=round(sum(c["seconds"] for c in calls), 2))

    def latency_summary(self) -> dict:
        """Per-purpose totals for the whole session (also stored in the report)."""
        by: dict[str, dict] = {}
        for c in self._llm_calls:
            b = by.setdefault(c["label"], {"calls": 0, "seconds": 0.0, "max_s": 0.0, "rate_limit_hits": 0,
                                           "sleep_s": 0.0, "completion_tokens": 0, "failed": 0})
            b["calls"] += 1
            b["seconds"] = round(b["seconds"] + c["seconds"], 2)
            b["max_s"] = max(b["max_s"], c["seconds"])
            b["rate_limit_hits"] += c.get("rate_limit_hits", 0)
            b["sleep_s"] = round(b["sleep_s"] + c.get("sleep_s", 0.0), 2)
            b["completion_tokens"] += c.get("completion_tokens", 0)
            b["failed"] += 0 if c["ok"] else 1
        return {"llm_calls": len(self._llm_calls), "llm_seconds": round(sum(c["seconds"] for c in self._llm_calls), 2),
                "by_purpose": by}

    def _flush(self) -> list[dict]:
        self._collect_llm_timing()
        out, self._turn_log = self._turn_log, []
        self.tool_log.extend(out)
        return out


def render_result(result, state: PatientState) -> str:
    """User-facing text for a finished assessment. Built only from JSON wording and rule output."""
    protocol = get_protocol(result.protocol_id)
    if result.urgency == "emergency":
        lines = [pathways.emergency_message(result.red_flag_category), "",
                 "Why: " + "; ".join(h.label for h in result.red_flags)]
    else:
        info = pathways.describe(result.urgency)
        lines = [f"URGENCY: {info['label']}", f"SUGGESTED CARE: {info['pathway']}", f"WHEN: {info['timeframe']}",
                 f"CONFIDENCE: {result.confidence}", "", info["advice"], "", "Why this level:"]
        lines += [f" - {r}" for r in result.reasons]
        lines += ["", "Seek help sooner if you notice:"] + [f" - {w}" for w in protocol.warning_signs]
        if result.needs_clinician_review:
            lines += ["", pathways.clinician_review_note()] + [f" - {r}" for r in result.clinician_reasons]
    lines += ["", pathways.disclaimer(), f"Your anonymous patient code: {state.patient_id}"]
    return "\n".join(lines)
