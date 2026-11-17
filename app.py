from __future__ import annotations

import httpx
import streamlit as st
from supabase import create_client

import config
from engine import pathways


st.set_page_config(
    page_title="TriageGuide",
    page_icon="\U0001FA7A",
    layout="centered",
)

supabase_auth = create_client(
    config.SUPABASE_URL,
    config.SUPABASE_PUBLISHABLE_KEY,
)

URGENCY_STYLE = {
    "emergency": "error",
    "urgent": "warning",
    "moderate": "info",
    "low": "success",
}


def show_api_error(exc: Exception) -> None:
    st.error(str(exc))


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

    if response.status_code >= 400:
        try:
            detail = response.json().get(
                "detail",
                response.text,
            )
        except ValueError:
            detail = response.text

        raise RuntimeError(
            f"API error {response.status_code}: {detail}"
        )

    return response.json()


def api_request_bytes(
    method: str,
    path: str,
) -> tuple[bytes, str]:
    token = st.session_state.get("auth_token")

    if not token:
        raise RuntimeError("You must log in first.")

    response = httpx.request(
        method,
        f"{config.API_URL}{path}",
        headers={
            "Authorization": f"Bearer {token}",
        },
        timeout=30,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"API error {response.status_code}: "
            f"{response.text}"
        )

    return (
        response.content,
        response.headers.get(
            "content-type",
            "text/plain",
        ),
    )


def load_current_user() -> dict:
    return api_request("GET", "/auth/me")


def clear_session() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def auth_page() -> None:
    st.title("\U0001FA7A TriageGuide")
    st.caption(
        "Please log in or create a patient account to continue."
    )

    mode = st.radio(
        "Account action",
        ["Log in", "Create patient account"],
        horizontal=True,
    )

    email = st.text_input("Email")
    password = st.text_input(
        "Password",
        type="password",
    )

    display_name = ""

    if mode == "Create patient account":
        display_name = st.text_input("Display name")

    if st.button(mode, type="primary"):
        if not email or not password:
            st.error(
                "Email and password are required."
            )
            return

        try:
            if mode == "Log in":
                response = (
                    supabase_auth.auth.sign_in_with_password({
                        "email": email,
                        "password": password,
                    })
                )
            else:
                response = supabase_auth.auth.sign_up({
                    "email": email,
                    "password": password,
                    "options": {
                        "data": {
                            "display_name": display_name,
                        },
                    },
                })

            if response.session is None:
                st.success(
                    "Account created. Check your email, "
                    "confirm it, then log in."
                )
                return

            st.session_state.auth_token = (
                response.session.access_token
            )
            st.session_state.auth_user = response.user
            st.session_state.app_user = load_current_user()

            st.rerun()

        except Exception as exc:
            st.error(
                f"Authentication failed: {exc}"
            )


def logout() -> None:
    try:
        supabase_auth.auth.sign_out()
    finally:
        clear_session()
        st.rerun()


def banner(urgency: str, text: str) -> None:
    getattr(
        st,
        URGENCY_STYLE.get(urgency, "info"),
    )(text)


def start_assessment() -> None:
    st.title("\U0001FA7A TriageGuide")
    st.caption(pathways.disclaimer())
    st.warning(
        f"If this is an emergency, call "
        f"{config.EMERGENCY_NUMBER} now."
    )

    if st.button(
        "Start assessment",
        type="primary",
    ):
        try:
            result = api_request(
                "POST",
                "/patient/assessments",
            )
        except Exception as exc:
            show_api_error(exc)
            return

        st.session_state.session_id = result["session_id"]
        st.session_state.patient_code = result["patient_code"]
        st.session_state.transcript = [{
            "role": "assistant",
            "text": result["message"],
            "kind": result["kind"],
        }]
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
        f"**{urgency.upper()}**: "
        f"{details['pathway']} "
        f"({details['timeframe']})",
    )

    if result.get("refusal"):
        st.info(result["refusal"])

    st.write(details["advice"])

    if not report:
        return

    st.caption(
        f"Confidence: "
        f"{report.get('confidence', 'unknown')}"
    )

    st.markdown("**Why this level**")

    for reason in report.get("reasons", []):
        st.markdown(f"- {reason}")

    if report.get("needs_clinician_review"):
        st.warning(
            pathways.clinician_review_note()
        )

    tool_log = report.get("tool_log", [])

    if tool_log:
        with st.expander(
            "How did the assistant decide?"
        ):
            st.json(tool_log)

    latency = report.get("latency")

    if latency:
        with st.expander("Response timing"):
            st.json(latency)


def show_patient_download() -> None:
    try:
        report_bytes, report_type = api_request_bytes(
            "GET",
            (
                "/patient/assessments/"
                f"{st.session_state.session_id}/download"
            ),
        )

        st.download_button(
            "Download summary for your doctor",
            data=report_bytes,
            file_name="triage-summary.txt",
            mime=report_type,
        )
    except Exception as exc:
        show_api_error(exc)


def patient_page() -> None:
    if "session_id" not in st.session_state:
        start_assessment()
        return

    st.title("\U0001FA7A TriageGuide")

    for message in st.session_state.get(
        "transcript",
        [],
    ):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.text(message["text"])
            else:
                st.write(message["text"])

    final = st.session_state.get("final")

    if final:
        st.divider()
        show_result(final)

        if final.get("kind") != "emergency":
            show_patient_download()

        st.info(
            "Your anonymous patient code: "
            f"**{st.session_state.patient_code}**"
        )

        if st.button("Delete my data"):
            try:
                api_request(
                    "DELETE",
                    "/patient/me",
                )
                clear_session()
                st.success(
                    "Your account data was deleted."
                )
                st.rerun()
            except Exception as exc:
                show_api_error(exc)

        return

    text = st.chat_input("Type your answer...")

    if not text:
        return

    st.session_state.transcript.append({
        "role": "user",
        "text": text,
        "kind": "user",
    })

    try:
        result = api_request(
            "POST",
            (
                "/patient/assessments/"
                f"{st.session_state.session_id}/messages"
            ),
            payload={"text": text},
        )
    except Exception as exc:
        show_api_error(exc)
        return

    if result["done"]:
        st.session_state.final = result
        assistant_text = (
            result["message"]
            if result["kind"] == "emergency"
            else (
                "Thank you. Here is your assessment."
            )
        )
    else:
        assistant_text = result["message"]

    st.session_state.transcript.append({
        "role": "assistant",
        "text": assistant_text,
        "kind": result["kind"],
    })

    st.rerun()


def dashboard_page() -> None:
    st.title("Clinician dashboard")

    user = st.session_state.get(
        "app_user",
        {},
    )

    if user.get("role") not in {
        "clinician",
        "reviewer",
        "admin",
    }:
        st.error("Clinician access required.")
        return

    try:
        data = api_request(
            "GET",
            "/clinician/sessions",
        )
    except Exception as exc:
        show_api_error(exc)
        return

    sessions = data.get("sessions", [])

    if not sessions:
        st.info("No sessions yet.")
        return

    counts = {
        urgency: sum(
            1
            for session in sessions
            if session.get("urgency") == urgency
        )
        for urgency in (
            "emergency",
            "urgent",
            "moderate",
            "low",
        )
    }

    columns = st.columns(4)

    for column, (name, count) in zip(
        columns,
        counts.items(),
    ):
        column.metric(
            name.capitalize(),
            count,
        )

    rows = [
        {
            "urgency": (
                session.get("urgency")
                or "(unfinished)"
            ),
            "patient": (
                session.get("patient_id")
                or ""
            ),
            "complaint": (
                session.get("complaint")
                or ""
            ),
            "session": (
                session.get("session_id")
                or ""
            ),
        }
        for session in sessions
    ]

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )

    session_ids = [
        row["session"]
        for row in rows
        if row["session"]
    ]

    if not session_ids:
        return

    selected = st.selectbox(
        "Open a session",
        session_ids,
    )

    try:
        data = api_request(
            "GET",
            f"/clinician/sessions/{selected}",
        )
    except Exception as exc:
        show_api_error(exc)
        return

    summary = data.get("summary") or {}

    if summary.get("urgency"):
        banner(
            summary["urgency"],
            f"**{summary['urgency_label']}**: "
            f"{summary['suggested_care']} "
            f"({summary['timeframe']})",
        )

    st.json(summary)

    if data.get("timeline"):
        with st.expander("Symptom timeline"):
            st.write(data["timeline"])

    if data.get("continuity"):
        with st.expander("Continuity note"):
            st.write(data["continuity"])

    report = data.get("report") or {}

    if report.get("tool_log"):
        with st.expander("Decision trace"):
            st.json(report["tool_log"])

    if report.get("latency"):
        with st.expander("LLM latency"):
            st.json(report["latency"])

    if st.button("Delete this patient's data"):
        try:
            patient_code = summary.get("patient_code")

            if not patient_code:
                st.error(
                    "Patient code is unavailable."
                )
                return

            api_request(
                "DELETE",
                (
                    "/clinician/patients/"
                    f"{patient_code}"
                ),
            )

            st.success("Patient data deleted.")
            st.rerun()

        except Exception as exc:
            show_api_error(exc)


if "auth_token" not in st.session_state:
    auth_page()
    st.stop()

if "app_user" not in st.session_state:
    try:
        st.session_state.app_user = (
            load_current_user()
        )
    except Exception:
        clear_session()
        st.rerun()

st.sidebar.caption(
    "Signed in as: "
    f"{st.session_state.app_user.get('email', 'user')}"
)

if st.sidebar.button("Log out"):
    logout()

role = st.session_state.app_user.get("role")

if role == "patient":
    page = st.sidebar.radio(
        "Page",
        ["Patient"],
    )
else:
    page = st.sidebar.radio(
        "Page",
        ["Clinician dashboard"],
    )

st.sidebar.caption(
    "Educational prototype. "
    "Not a medical device. Never diagnoses."
)

if page == "Patient":
    patient_page()
else:
    dashboard_page()