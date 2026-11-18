from __future__ import annotations
from urllib.parse import quote
import html
import json

import httpx
import streamlit as st
from supabase import create_client
from agent.workflow_visualization import workflow_diagram
import config
from engine import pathways
import streamlit.components.v1 as components


st.set_page_config(
    page_title="TriageGuide",
    page_icon="\U0001FA7A",
    layout="wide",
    initial_sidebar_state="expanded",
)

URGENCY_STYLE = {
    "emergency": "error",
    "urgent": "warning",
    "moderate": "info",
    "low": "success",
}

URGENCY_ICON = {
    "emergency": "🚨",
    "urgent": "⚠️",
    "moderate": "🩹",
    "low": "✅",
}

CLINICIAN_ROLES = {"clinician", "reviewer", "admin"}

FEEDBACK_OPTIONS = {
    "✓ Correct": "correct",
    "⚠ Needs review": "needs_review",
    "↕ Incorrect urgency": "incorrect_urgency",
}


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f4f7fb; }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { padding-top: 2rem; max-width: 1200px; }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #102a43 0%, #243b53 100%);
        }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 { color: #ffffff; }
        [data-testid="stSidebar"] .stButton button {
            background: #ffffff;
            border: none;
            border-radius: 10px;
            font-weight: 600;
        }
        [data-testid="stSidebar"] .stButton button p { color: #102a43; }

        /* Hero header */
        .hero {
            background: linear-gradient(135deg, #1f6feb 0%, #12b5a6 100%);
            border-radius: 18px;
            padding: 26px 30px;
            margin-bottom: 22px;
            box-shadow: 0 8px 24px rgba(31, 111, 235, 0.25);
        }
        .hero h2 { margin: 0; color: #ffffff; font-weight: 800; }
        .hero p  { margin: 6px 0 0 0; color: rgba(255,255,255,0.88); }

        /* Cards */
        .triage-card {
            background: #ffffff;
            border: 1px solid #e6ebf2;
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 12px;
            box-shadow: 0 3px 10px rgba(16, 42, 67, 0.05);
        }
        .triage-card h4 {
            margin: 0 0 6px 0;
            color: #627d98;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: .05em;
        }
        .card-value { font-size: 26px; font-weight: 800; color: #102a43; }
        .urgency-emergency { border-left: 7px solid #d64545; }
        .urgency-urgent    { border-left: 7px solid #f0a020; }
        .urgency-moderate  { border-left: 7px solid #3a86c7; }
        .urgency-low       { border-left: 7px solid #36a269; }

        /* Chat + expanders */
        div[data-testid="stChatMessage"] {
            border: 1px solid #e6ebf2;
            border-radius: 14px;
            background: #ffffff;
            margin-bottom: 10px;
            padding: 12px;
            box-shadow: 0 2px 6px rgba(16, 42, 67, 0.04);
        }
        div[data-testid="stExpander"] {
            border: 1px solid #e6ebf2;
            border-radius: 10px;
            background: #ffffff;
            margin-bottom: 8px;
        }

        /* Buttons */
        .stButton button[kind="primary"],
        [data-testid="stFormSubmitButton"] button[kind="primary"] {
            border-radius: 10px;
            font-weight: 700;
        }

        /* Progress steps */
        .steps { display: flex; gap: 8px; margin: 4px 0 18px 0; }
        .step {
            flex: 1;
            text-align: center;
            padding: 8px 4px;
            border-radius: 999px;
            font-size: 13px;
            background: #e9eef5;
            color: #829ab1;
        }
        .step-active   { background: #e6f2ff; color: #175cd3; font-weight: 700; }
        .step-complete { background: #dff4e8; color: #18794e; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h2>🩺 {html.escape(title)}</h2>
            <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_card(
    title: str,
    value: str,
    urgency: str | None = None,
) -> None:
    css_class = f"urgency-{urgency}" if urgency else ""

    st.markdown(
        f"""
        <div class="triage-card {css_class}">
            <h4>{html.escape(title)}</h4>
            <div class="card-value">{html.escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_assessment_progress(current_step: int) -> None:
    labels = ["Symptoms", "Questions", "Red flags", "Urgency", "Summary"]
    parts = []

    for index, label in enumerate(labels, start=1):
        if index < current_step:
            css, marker = "step-complete", "✓"
        elif index == current_step:
            css, marker = "step-active", "•"
        else:
            css, marker = "", "○"

        parts.append(f'<div class="step {css}">{marker} {label}</div>')

    st.markdown(
        f'<div class="steps">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------

def get_auth_client():
    # Created per call on purpose: a shared module-level client would hold
    # one user's session for every visitor of the server.
    return create_client(
        config.SUPABASE_URL,
        config.SUPABASE_PUBLISHABLE_KEY,
    )


def show_api_error(exc: Exception) -> None:
    st.error(str(exc))


def clear_session(flash: str | None = None) -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    if flash:
        st.session_state["flash"] = flash


def _check_response(response: httpx.Response) -> None:
    if response.status_code == 401:
        clear_session("Your session expired. Please log in again.")
        st.rerun()

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text

        raise RuntimeError(f"API error {response.status_code}: {detail}")


def api_request(
    method: str,
    path: str,
    payload: dict | None = None,
) -> dict:
    token = st.session_state.get("auth_token")

    if not token:
        raise RuntimeError("You must log in first.")

    response = httpx.request(
        method,
        f"{config.API_URL}{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )

    _check_response(response)
    return response.json()


def api_request_bytes(method: str, path: str) -> tuple[bytes, str]:
    token = st.session_state.get("auth_token")

    if not token:
        raise RuntimeError("You must log in first.")

    response = httpx.request(
        method,
        f"{config.API_URL}{path}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )

    _check_response(response)

    return (
        response.content,
        response.headers.get("content-type", "text/plain"),
    )


def load_current_user() -> dict:
    return api_request("GET", "/auth/me")


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

def handle_auth(
    mode: str,
    email: str,
    password: str,
    display_name: str = "",
) -> None:
    if not email or not password:
        st.error("Email and password are required.")
        return

    client = get_auth_client()

    try:
        if mode == "login":
            response = client.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
        else:
            response = client.auth.sign_up(
                {
                    "email": email,
                    "password": password,
                    "options": {"data": {"display_name": display_name}},
                }
            )

        if response.session is None:
            st.success(
                "Account created. Check your email, confirm it, then log in."
            )
            return

        st.session_state.auth_token = response.session.access_token
        st.session_state.auth_user = response.user
        st.session_state.app_user = load_current_user()

        st.rerun()

    except Exception as exc:
        st.error(f"Authentication failed: {exc}")


def auth_page() -> None:
    _, middle, _ = st.columns([1, 1.6, 1])

    with middle:
        render_app_header(
            "TriageGuide",
            "Safety-focused health assessment. Log in or create a patient account.",
        )

        login_tab, signup_tab = st.tabs(["Log in", "Create patient account"])

        with login_tab:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button(
                    "Log in",
                    type="primary",
                    width="stretch",
                )

            if submitted:
                handle_auth("login", email, password)

        with signup_tab:
            with st.form("signup_form"):
                display_name = st.text_input("Display name")
                email = st.text_input("Email", key="signup_email")
                password = st.text_input(
                    "Password",
                    type="password",
                    key="signup_password",
                )
                submitted = st.form_submit_button(
                    "Create patient account",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                handle_auth("signup", email, password, display_name)


def logout() -> None:
    clear_session("You have been logged out.")
    st.rerun()


# --------------------------------------------------------------------------
# Shared widgets
# --------------------------------------------------------------------------

def banner(urgency: str, text: str) -> None:
    getattr(st, URGENCY_STYLE.get(urgency, "info"))(text)


def show_explainability_panel(
    state: dict,
    report: dict,
    summary: dict,
    workflow: dict,
) -> None:
    with st.expander("🔍 Explainability", expanded=True):
        left, right = st.columns(2)

        with left:
            st.markdown("**Confidence**")
            st.write(report.get("confidence", "unknown"))

            if report.get("needs_clinician_review"):
                st.warning(pathways.clinician_review_note())

        with right:
            reasons = report.get("reasons", [])
            st.markdown("**Why this urgency level**")

            if reasons:
                for reason in reasons:
                    st.markdown(f"- {reason}")
            else:
                st.caption("No reasons recorded.")

        if state:
            with st.expander("Session state"):
                st.json(state)


# --------------------------------------------------------------------------
# Patient
# --------------------------------------------------------------------------

def start_assessment() -> None:
    st.caption(pathways.disclaimer())
    st.warning(f"If this is an emergency, call {config.EMERGENCY_NUMBER} now.")

    if st.button("Start assessment", type="primary"):
        try:
            result = api_request("POST", "/patient/assessments")
        except Exception as exc:
            show_api_error(exc)
            return

        st.session_state.session_id = result["session_id"]
        st.session_state.patient_code = result["patient_code"]
        st.session_state.transcript = [
            {
                "role": "assistant",
                "text": result["message"],
                "kind": result["kind"],
            }
        ]
        st.session_state.final = None
        st.rerun()


def show_result(result: dict) -> None:
    report = result.get("result") or {}

    if result.get("kind") == "emergency":
        st.error(result["message"])
        return

    urgency = result.get("urgency")

    if not urgency:
        st.info(result["message"])
        return

    details = pathways.describe(urgency)

    banner(
        urgency,
        f"{URGENCY_ICON.get(urgency, '')} **{urgency.upper()}**: "
        f"{details['pathway']} ({details['timeframe']})",
    )

    if result.get("refusal"):
        st.info(result["refusal"])

    st.write(details["advice"])

    if not report:
        return

    st.caption(f"Confidence: {report.get('confidence', 'unknown')}")

    st.markdown("**Why this level**")

    for reason in report.get("reasons", []):
        st.markdown(f"- {reason}")

    if report.get("needs_clinician_review"):
        st.warning(pathways.clinician_review_note())

    if report.get("tool_log"):
        with st.expander("How did the assistant decide?"):
            st.json(report["tool_log"])

    if report.get("latency"):
        with st.expander("Response timing"):
            st.json(report["latency"])


def show_patient_download() -> None:
    try:
        report_bytes, report_type = api_request_bytes(
            "GET",
            f"/patient/assessments/{st.session_state.session_id}/download",
        )

        st.download_button(
            "📄 Download summary for your doctor",
            data=report_bytes,
            file_name="triage-summary.txt",
            mime=report_type,
            width="stretch",
        )
    except Exception as exc:
        show_api_error(exc)


def patient_page() -> None:
    if "session_id" not in st.session_state:
        start_assessment()
        return

    transcript = st.session_state.get("transcript", [])
    final = st.session_state.get("final")

    user_messages = sum(1 for m in transcript if m["role"] == "user")

    if final:
        step = 5
    else:
        step = min(1 + user_messages, 3)  # cosmetic estimate

    render_assessment_progress(step)

    for message in transcript:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.text(message["text"])
            else:
                st.write(message["text"])

    if final:
        st.divider()
        show_result(final)

        if final.get("kind") != "emergency":
            show_patient_download()

        st.info(
            "Your anonymous patient code: "
            f"**{st.session_state.patient_code}**"
        )

        if st.button("🗑 Delete my data", width="stretch"):
            try:
                api_request("DELETE", "/patient/me")
            except Exception as exc:
                show_api_error(exc)
                return

            clear_session("Your account data was deleted.")
            st.rerun()

        return

    text = st.chat_input("Type your answer...")

    if not text:
        return

    st.session_state.transcript.append(
        {"role": "user", "text": text, "kind": "user"}
    )

    with st.chat_message("user"):
        st.text(text)

    try:
        with st.spinner("Analyzing the response..."):
            result = api_request(
                "POST",
                f"/patient/assessments/{st.session_state.session_id}/messages",
                payload={"text": text},
            )
    except Exception as exc:
        show_api_error(exc)
        return

    if result["done"]:
        st.session_state.final = result
        assistant_text = (
            ""
            if result["kind"] == "emergency"
            else "Thank you. Here is your assessment."
        )
    else:
        assistant_text = result["message"]

    if assistant_text:
        st.session_state.transcript.append(
            {
                "role": "assistant",
                "text": assistant_text,
                "kind": result["kind"],
            }
        )

    st.rerun()


# --------------------------------------------------------------------------
# Clinician dashboard
# --------------------------------------------------------------------------

def render_workflow(diagram: str) -> None:
    safe_diagram = json.dumps(diagram).replace("</", "<\\/")

    workflow_html = f"""
    <style>
    html, body {{ margin: 0; }}
    #workflow {{
        width: 100%;
        background: white;
        font-family: sans-serif;
    }}
    #workflow svg {{ max-width: 100%; height: auto; }}
    #workflow-error {{ color: #b42318; }}
    </style>

    <div id="workflow">Loading diagram...</div>

    <script type="module">
      import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";

      mermaid.initialize({{
        startOnLoad: false,
        securityLevel: "strict",
        theme: "neutral",
        flowchart: {{ useMaxWidth: true, htmlLabels: true }}
      }});

      try {{
        const diagram = {safe_diagram};
        const renderId = "wf-" + Math.random().toString(36).slice(2);
        const {{ svg }} = await mermaid.render(renderId, diagram);
        document.getElementById("workflow").innerHTML = svg;
      }} catch (error) {{
        document.getElementById("workflow").innerHTML =
          '<p id="workflow-error">Unable to render workflow diagram.</p>';
        console.error(error);
      }}
    </script>
    """

    st.caption("🟩 executed   🟨 flagged for review   🟥 emergency   ⬜ not reached")

    components.html(
        workflow_html,
        height=760,
        scrolling=True,
    )

def dashboard_page() -> None:
    user = st.session_state.get("app_user", {})

    if user.get("role") not in CLINICIAN_ROLES:
        st.error("Clinician access required.")
        return

    try:
        data = api_request("GET", "/clinician/sessions")
    except Exception as exc:
        show_api_error(exc)
        return

    sessions = data.get("sessions", [])

    if not sessions:
        st.info("No sessions yet.")
        return

    # --- Overview metrics -------------------------------------------------
    urgencies = ("emergency", "urgent", "moderate", "low")
    columns = st.columns(4)

    for column, name in zip(columns, urgencies):
        count = sum(1 for s in sessions if s.get("urgency") == name)

        with column:
            render_status_card(
                f"{URGENCY_ICON[name]} {name.capitalize()}",
                str(count),
                urgency=name,
            )

    rows = [
        {
            "urgency": s.get("urgency") or "(unfinished)",
            "patient": s.get("patient_id") or "",
            "complaint": s.get("complaint") or "",
            "session": s.get("session_id") or "",
        }
        for s in sessions
    ]

    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
    )

    labels = {
        row["session"]: (
            f"{row['urgency']} · {row['complaint'] or 'no complaint'} "
            f"· {row['session'][:8]}"
        )
        for row in rows
        if row["session"]
    }

    if not labels:
        return

    selected = st.selectbox(
        "Open a session",
        list(labels),
        format_func=lambda sid: labels[sid],
    )

    try:
        detail = api_request("GET", f"/clinician/sessions/{selected}")
    except Exception as exc:
        show_api_error(exc)
        return

    summary = detail.get("summary") or {}
    state = detail.get("state") or {}
    report = detail.get("report") or {}
    workflow = detail.get("workflow") or {}

    st.divider()

    if summary.get("urgency"):
        banner(
            summary["urgency"],
            f"{URGENCY_ICON.get(summary['urgency'], '')} "
            f"**{summary.get('urgency_label', summary['urgency'].upper())}**: "
            f"{summary.get('suggested_care', '')} "
            f"({summary.get('timeframe', '')})",
        )

    overview_tab, explain_tab, trace_tab, workflow_tab, feedback_tab = st.tabs(
        ["Overview", "Explainability", "Trace", "Workflow", "Feedback"]
    )

    with overview_tab:
        st.json(summary)

        if detail.get("timeline"):
            with st.expander("Symptom timeline"):
                st.write(detail["timeline"])

        if detail.get("continuity"):
            with st.expander("Continuity note"):
                st.write(detail["continuity"])

        try:
            pdf_bytes, pdf_type = api_request_bytes(
                "GET",
                f"/clinician/sessions/{selected}/pdf",
            )
            st.download_button(
                "📄 Download clinician PDF",
                data=pdf_bytes,
                file_name=f"triage-{selected}.pdf",
                mime=pdf_type,
                width="stretch",
            )
        except Exception as exc:
            show_api_error(exc)

    with explain_tab:
        show_explainability_panel(state, report, summary, workflow)

    with trace_tab:
        if report.get("tool_log"):
            with st.expander("Decision trace", expanded=True):
                st.json(report["tool_log"])

        if report.get("latency"):
            with st.expander("LLM latency"):
                st.json(report["latency"])

        if not report.get("tool_log") and not report.get("latency"):
            st.caption("No trace data for this session.")

    with workflow_tab:
        if workflow.get("diagram"):
            render_workflow(workflow["diagram"])
        else:
            st.caption("No workflow diagram for this session.")

    with feedback_tab:
        previous = detail.get("feedback")

        if previous:
            with st.expander("Existing feedback"):
                st.json(previous)

        with st.form(f"feedback_form_{selected}"):
            choice = st.radio(
                "Clinician assessment",
                list(FEEDBACK_OPTIONS),
                horizontal=True,
            )
            comment = st.text_area("Comment (optional)")
            submitted = st.form_submit_button(
                "Save clinician feedback",
                type="primary",
            )

        if submitted:
            try:
                outcome = FEEDBACK_OPTIONS[choice]

                payload = {
                    "outcome": outcome,
                    "comment": comment.strip() or None,
                }

                if outcome == "incorrect_urgency":
                    corrected = st.selectbox(
                        "Correct urgency",
                        ["low", "moderate", "urgent", "emergency"],
                    )
                    payload["corrected_urgency"] = corrected

                api_request(
                    "POST",
                    f"/clinician/sessions/{selected}/feedback",
                    payload=payload,
                )
                st.success("Feedback saved.")
            except Exception as exc:
                show_api_error(exc)

        st.divider()
        st.markdown("**Danger zone**")

        confirm = st.checkbox(
            "I understand this permanently deletes this patient's data",
            key=f"confirm_delete_{selected}",
        )

        if st.button("🗑 Delete this patient's data", disabled=not confirm):
            patient_code = summary.get("patient_code")

            if not patient_code:
                st.error("Patient code is unavailable.")
            else:
                try:
                    api_request(
                        "DELETE",
                        f"/clinician/patients/{patient_code}",
                    )
                    st.session_state["flash"] = "Patient data deleted."
                    st.rerun()
                except Exception as exc:
                    show_api_error(exc)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

inject_css()

flash = st.session_state.pop("flash", None)

if "auth_token" not in st.session_state:
    if flash:
        st.info(flash)
    auth_page()
    st.stop()

if "app_user" not in st.session_state:
    try:
        st.session_state.app_user = load_current_user()
    except Exception:
        clear_session("Could not load your account. Please log in again.")
        st.rerun()

if flash:
    st.success(flash)

role = st.session_state.app_user.get("role")

st.sidebar.markdown("### 🩺 TriageGuide")
st.sidebar.caption(
    f"Signed in as: {st.session_state.app_user.get('email', 'user')}"
)

if st.sidebar.button("Log out", use_container_width=True):
    logout()

if role == "patient":
    page = st.sidebar.radio("Page", ["Patient"])
    render_app_header(
        "TriageGuide",
        "Safety-focused health assessment",
    )
else:
    page = st.sidebar.radio("Page", ["Clinician dashboard"])
    render_app_header(
        "Clinician dashboard",
        "Review sessions, explainability and feedback",
    )

st.sidebar.caption(
    "Educational prototype. Not a medical device. Never diagnoses."
)

if page == "Patient":
    patient_page()
else:
    dashboard_page()


