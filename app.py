"""TriageGuide web app (Streamlit): patient chat + clinician dashboard.

Run from the project root:   streamlit run app.py
Educational decision-support prototype. It never diagnoses and never recommends medicines.
"""
from __future__ import annotations

import streamlit as st

import config
from agent.llm_client import get_llm
from agent.orchestrator import TriageAgent
from agent.summary import build_clinician_summary, format_tool_log, to_markdown
from engine import pathways
from engine.protocols import get_protocol
from memory.db import Memory

st.set_page_config(page_title="TriageGuide", page_icon="\U0001FA7A", layout="centered")

URGENCY_STYLE = {"emergency": "error", "urgent": "warning", "moderate": "info", "low": "success"}


@st.cache_resource
def get_memory() -> Memory:
    return Memory()


def make_llm(use_llm: bool):
    """(llm or None, message). Never raises: the app still works without an LLM."""
    if not use_llm:
        return None, "AI model switched off: fixed question order."
    try:
        return get_llm(), ""
    except Exception as exc:
        return None, f"AI model unavailable ({exc}). Using the fixed question order."


def banner(urgency: str, text: str) -> None:
    getattr(st, URGENCY_STYLE.get(urgency, "info"))(text)


# ======================================================================= patient page
def start_form() -> None:
    st.title("\U0001FA7A TriageGuide")
    st.caption(pathways.disclaimer())
    st.warning(f"If this is an emergency, call {config.EMERGENCY_NUMBER} now.")
    code = st.text_input("Patient code from an earlier visit (optional)", placeholder="P-XXXXXX").strip().upper()
    has_key = bool(config.GROQ_API_KEY) or config.LLM_PROVIDER == "gemini"
    use_llm = st.checkbox("Use the AI model for questions", value=has_key,
                          help="Off = fixed question order (works without internet or a key).")
    if st.button("Start assessment", type="primary"):
        mem = get_memory()
        llm, note = make_llm(use_llm)
        agent = TriageAgent(mem, llm)
        notice = note
        try:
            turn = agent.start(code or None)
        except KeyError:
            turn = agent.start(None)
            notice = (notice + " " if notice else "") + "That patient code was not found, so a new record was started."
        st.session_state.agent = agent
        st.session_state.transcript = [{"role": "assistant", "text": turn.message, "kind": turn.kind}]
        st.session_state.notice = notice
        st.session_state.final = None
        st.rerun()


def show_result(turn) -> None:
    report = turn.result
    if turn.kind == "emergency":
        st.error(turn.message)
        return
    banner(turn.urgency, f"**{report['urgency'].upper()}**: "
                         f"{pathways.describe(turn.urgency)['pathway']} ({pathways.describe(turn.urgency)['timeframe']})")
    if turn.refusal:
        st.info(turn.refusal)
    st.write(pathways.describe(turn.urgency)["advice"])
    st.caption(f"Confidence: {report['confidence']}")
    st.markdown("**Why this level**")
    for reason in report["reasons"]:
        st.markdown(f"- {reason}")
    protocol = get_protocol(report["protocol_id"])
    st.markdown("**Seek help sooner if you notice**")
    for sign in protocol.warning_signs:
        st.markdown(f"- {sign}")
    if report["needs_clinician_review"]:
        st.warning(pathways.clinician_review_note())


def patient_page() -> None:
    if "agent" not in st.session_state:
        start_form()
        return
    agent: TriageAgent = st.session_state.agent
    mem = get_memory()
    st.title("\U0001FA7A TriageGuide")
    if st.session_state.get("notice"):
        st.caption(st.session_state.notice)

    for m in st.session_state.transcript:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.text(m["text"])          # plain text: patient input is never rendered as markdown
            elif m["kind"] == "emergency":
                st.error(m["text"])
            else:
                st.write(m["text"])

    if agent.done:
        final = st.session_state.final
        st.divider()
        show_result(final)
        st.info(f"Your anonymous patient code: **{agent.state.patient_id}**. "
                "Enter it next time so the system can remember this visit. No name or phone number is stored.")
        summary = build_clinician_summary(agent.state, final.result, mem.timeline_text(agent.state.patient_id))
        st.download_button("Download summary for your doctor", to_markdown(summary),
                           file_name=f"triage_{agent.state.patient_id}.md", mime="text/markdown")
        with st.expander("How did the assistant decide? (tool calls)"):
            for line in format_tool_log(agent.tool_log):
                st.text(line)
        c1, c2 = st.columns(2)
        if c1.button("Start a new assessment"):
            for k in ("agent", "transcript", "final", "notice"):
                st.session_state.pop(k, None)
            st.rerun()
        if c2.button("Delete my data"):
            mem.delete_patient(agent.state.patient_id)
            for k in ("agent", "transcript", "final", "notice"):
                st.session_state.pop(k, None)
            st.success("Your record and all linked sessions were deleted.")
        return

    text = st.chat_input("Type your answer...")
    if text:
        st.session_state.transcript.append({"role": "user", "text": text, "kind": "user"})
        with st.spinner("Thinking..."):
            turn = agent.handle(text)
        if turn.done:
            st.session_state.final = turn
            st.session_state.transcript.append({"role": "assistant", "kind": turn.kind,
                                                "text": turn.message if turn.kind == "emergency"
                                                else "Thank you. Here is my assessment."})
        else:
            st.session_state.transcript.append({"role": "assistant", "text": turn.message, "kind": turn.kind})
        st.rerun()


# ======================================================================= clinician dashboard
def dashboard_page() -> None:
    st.title("Clinician dashboard")
    if config.DASHBOARD_PIN:
        if not st.session_state.get("pin_ok"):
            pin = st.text_input("Dashboard PIN", type="password")
            if pin and pin == config.DASHBOARD_PIN:
                st.session_state.pin_ok = True
                st.rerun()
            elif pin:
                st.error("Wrong PIN.")
            return
    else:
        st.caption("No DASHBOARD_PIN is set: anyone who can open this page can see all sessions (OK for a local demo only).")

    mem = get_memory()
    sessions = mem.list_sessions(limit=200)
    if not sessions:
        st.info("No sessions yet. Run an assessment on the Patient page first.")
        return

    counts = {u: sum(1 for s in sessions if s["urgency"] == u) for u in ("emergency", "urgent", "moderate", "low")}
    cols = st.columns(4)
    for col, (name, n) in zip(cols, counts.items()):
        col.metric(name.capitalize(), n)

    only_open = st.checkbox("Only show sessions that recommend clinician review")
    show_unfinished = st.checkbox("Also show unfinished sessions", value=False)
    rows = []
    for s in sessions:
        report = mem.get_report(s["session_id"]) or {}
        if not report and not show_unfinished:
            continue
        if only_open and not report.get("needs_clinician_review"):
            continue
        rows.append({"urgency": s["urgency"] or "(unfinished)", "patient": s["patient_id"],
                     "complaint": s["complaint"] or "", "review": "yes" if report.get("needs_clinician_review") else "",
                     "when": s["created_at"][:16].replace("T", " "), "session": s["session_id"]})
    if not rows:
        st.info("No finished assessments match. Tick \"Also show unfinished sessions\" to see the rest.")
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)

    chosen = st.selectbox("Open a session", [r["session"] for r in rows],
                          format_func=lambda sid: next(f"{r['urgency']} | {r['patient']} | {r['complaint']} | {r['when']}"
                                                       for r in rows if r["session"] == sid))
    state = mem.load_state(chosen)
    report = mem.get_report(chosen)
    summary = build_clinician_summary(
        state, report, mem.timeline_text(state.patient_id),
        mem.continuity_note(state.patient_id, exclude_session_id=chosen))
    if summary["urgency"]:
        banner(summary["urgency"], f"**{summary['urgency_label']}**: {summary['suggested_care']} ({summary['timeframe']})")
    st.markdown(to_markdown(summary, base_level=3))
    st.download_button("Download this summary", to_markdown(summary), file_name=f"triage_{chosen}.md",
                       mime="text/markdown")
    if report and report.get("tool_log"):
        with st.expander("Agent decision trace"):
            for line in format_tool_log(report["tool_log"]):
                st.text(line)
        if report.get("latency"):
            with st.expander("LLM latency for this session"):
                st.json(report["latency"])
    if st.button("Delete this patient's data (all sessions)"):
        mem.delete_patient(state.patient_id)
        st.success("Deleted.")
        st.rerun()


# ======================================================================= main
page = st.sidebar.radio("Page", ["Patient", "Clinician dashboard"])
st.sidebar.caption("Educational prototype. Not a medical device. Never diagnoses.")
if page == "Patient":
    patient_page()
else:
    dashboard_page()
