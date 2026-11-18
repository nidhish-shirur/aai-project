# TriageGuide

TriageGuide is an educational healthcare decision-support application that collects patient symptoms, identifies safety concerns, applies deterministic triage rules, and suggests an appropriate care pathway.

> **Important:** TriageGuide is not a medical device. It does not diagnose diseases, prescribe medication, or replace a qualified healthcare professional. The rules and protocols are educational and have not been clinically validated.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Environment Configuration](#environment-configuration)
- [Running the Application](#running-the-application)
- [Running the FastAPI Backend](#running-the-fastapi-backend)
- [Supabase Setup](#supabase-setup)
- [Command-Line Tools](#command-line-tools)
- [Testing](#testing)
- [Evaluation](#evaluation)
- [Knowledge Base](#knowledge-base)
- [Safety Design](#safety-design)
- [Data Storage and Privacy](#data-storage-and-privacy)
- [API Overview](#api-overview)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Development Guidelines](#development-guidelines)
- [License and Attribution](#license-and-attribution)

---

## Overview

TriageGuide combines deterministic clinical-style decision rules with optional language-model capabilities.

The system:

1. Receives a patient's free-text description.
2. Checks the message for emergency red flags.
3. Detects the patient's language.
4. Extracts structured facts such as symptoms, duration, severity, age, pregnancy status, and risk factors.
5. Validates extracted information using Python code.
6. Asks follow-up questions based on the selected complaint protocol.
7. Calculates an urgency level using deterministic rules.
8. Optionally performs an escalation-only AI safety review.
9. Produces a care pathway and a clinician-facing summary.
10. Stores the assessment and symptom timeline for continuity.

The final urgency decision is controlled by deterministic Python rules. The language model cannot directly diagnose the patient or reduce a safety escalation.

---

## Features

### Patient features

- Free-text symptom description
- Deterministic or LLM-assisted questioning
- Support for English, Hindi, Hinglish, and other languages
- Emergency red-flag detection
- Urgency classification:
  - `low`
  - `moderate`
  - `urgent`
  - `emergency`
- Assessment progress indicator
- Anonymous patient codes
- Returning-patient continuity
- Symptom timeline
- Downloadable assessment summary
- Data deletion

### Clinician features

- Clinician dashboard with urgency counts
- Session list and session selector
- Patient assessment summaries
- Reported, denied, and unknown symptoms
- Red-flag information
- Suggested care pathway
- Explainability panel (confidence, reasons, clinician-review flag)
- Decision trace
- LangGraph workflow visualization that highlights the path each session took
- LLM latency information
- PDF report generation
- Clinician feedback:
  - Correct
  - Needs review
  - Incorrect urgency

### Engineering and evaluation features

- Deterministic safety gate
- Protocol-based questioning
- LLM extraction with validation
- Output guardrails
- Circuit breaker for repeated model failures
- SQLite local persistence
- Supabase persistence for the authenticated API
- FastAPI backend
- Streamlit interface
- Offline test suite
- Held-out evaluation cases
- Ablation testing
- Adversarial prompt testing
- Latency reporting

---

## System Architecture

```text
Patient message
      |
      v
Language detection
      |
      v
Deterministic text red-flag gate
      |
      +---- Emergency detected ----> Emergency guidance
      |
      v
Language agent (LLM)
      |
      v
Fact extraction
      |
      v
Fact validation and state update
      |
      v
Fact-level red-flag detection
      |
      +---- Emergency detected ----> Emergency guidance
      |
      v
Adaptive follow-up questioning
      |
      +---- Information missing ----> Wait for patient response
      |
      v
Medical knowledge (protocol lookup)
      |
      v
Deterministic urgency engine
      |
      v
Optional AI safety review (escalation only)
      |
      v
Clinician escalation check
      |
      v
Care pathway and clinician summary
      |
      v
SQLite or Supabase persistence
```

### Safety boundary

The LLM is used for language-related tasks such as:

- Extracting facts from free text
- Detecting language
- Generating a follow-up question
- Providing an escalation-only second opinion

The LLM is not trusted to:

- Decide the final urgency level
- Override deterministic emergency rules
- Diagnose a condition
- Recommend medication
- Invent facts
- Remove required safety questions

---

## Technology Stack

- Python 3.11+
- FastAPI
- Uvicorn
- Streamlit
- LangGraph
- Mermaid (workflow diagram rendering)
- Pydantic
- Groq API
- Optional Google Gemini client
- SQLite
- Supabase and PostgreSQL
- PyJWT
- HTTPX
- pytest
- pandas
- fpdf2

---

## Project Structure

```text
.
├── agent/
│   ├── extractor.py
│   ├── guard.py
│   ├── langgraph_workflow.py
│   ├── llm_client.py
│   ├── orchestrator.py
│   ├── safety_review.py
│   ├── state.py
│   ├── summary.py
│   └── workflow_visualization.py
│
├── api/
│   ├── auth.py
│   ├── dependencies.py
│   ├── main.py
│   ├── pdf_report.py
│   ├── rate_limit.py
│   └── schemas.py
│
├── data/
│   ├── prepare_vocab.py
│   ├── gold_vignettes.json
│   ├── dataset_report.txt
│   └── raw/
│
├── engine/
│   ├── answer_parser.py
│   ├── conditions.py
│   ├── escalation.py
│   ├── pathways.py
│   ├── protocols.py
│   ├── red_flags.py
│   ├── triage_rules.py
│   └── vocab.py
│
├── eval/
│   ├── ablations.py
│   ├── baselines.py
│   ├── extraction_cases.json
│   ├── extraction_eval.py
│   ├── heldout_cases.json
│   ├── latency_report.py
│   ├── metrics.py
│   ├── run_eval.py
│   ├── run_vignettes.py
│   └── vignette_runner.py
│
├── knowledge/
│   ├── common.json
│   ├── modifiers.json
│   ├── pathways.json
│   ├── red_flags.json
│   ├── synonyms.json
│   ├── symptom_vocab.csv
│   └── protocols/
│
├── memory/
│   ├── db.py
│   ├── repository.py
│   ├── supabase_repository.py
│   └── timeline.py
│
├── scripts/
│   ├── chat_cli.py
│   ├── check_knowledge.py
│   ├── cli_fallback.py
│   ├── demo_engine.py
│   ├── demo_memory.py
│   ├── migrate_sqlite_to_supabase.py
│   ├── test_adversarial.py
│   ├── test_extractor.py
│   └── test_llm.py
│
├── supabase/
│   └── migrations/
│       └── 001_initial_schema.sql
│
├── tests/
├── app.py
├── config.py
├── requirements.txt
├── pytest.ini
└── README.md
```

### Important modules

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit patient interface and clinician dashboard |
| `api/main.py` | FastAPI routes and backend application |
| `agent/langgraph_workflow.py` | LangGraph workflow orchestration |
| `agent/workflow_visualization.py` | Mermaid workflow diagram and executed-path highlighting |
| `agent/extractor.py` | Extracts structured facts from patient text |
| `agent/guard.py` | Rejects unsafe generated content |
| `agent/safety_review.py` | Escalation-only AI review |
| `agent/summary.py` | Creates deterministic clinician summaries |
| `engine/red_flags.py` | Detects emergency symptoms |
| `engine/triage_rules.py` | Computes urgency |
| `engine/protocols.py` | Loads and applies complaint protocols |
| `engine/pathways.py` | Provides care recommendations and urgency text |
| `memory/db.py` | Local SQLite persistence |
| `memory/supabase_repository.py` | Supabase persistence |
| `knowledge/protocols/` | Complaint-specific questions and rules |
| `tests/` | Automated tests |
| `eval/` | Evaluation and benchmarking tools |

---

## Requirements

- Python 3.11 or newer
- Git
- Internet connection for installing dependencies
- Windows, macOS, or Linux

Optional requirements:

- Groq API key for live LLM features
- Supabase project for the authenticated API
- Kaggle dataset files for rebuilding the symptom vocabulary

---

## Installation

### Windows PowerShell

```powershell
git clone <repository-url>
cd aai-project

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents environment activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Environment Configuration

Copy the example configuration:

### Windows

```powershell
copy .env.example .env
```

### macOS or Linux

```bash
cp .env.example .env
```

Open `.env` and configure the required values.

### LLM configuration

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

If the configured model is unavailable, check the current Groq model list and update `GROQ_MODEL`.

Optional Gemini configuration:

```dotenv
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.0-flash
```

### General configuration

```dotenv
EMERGENCY_NUMBER=112
MAX_QUESTIONS=15
LLM_TIMEOUT_S=12
DASHBOARD_PIN=
TRIAGE_DB=triageguide.db
```

### Supabase configuration

The FastAPI backend and the Streamlit login both require:

```dotenv
SUPABASE_URL=https://YOUR_PROJECT_ID.supabase.co
SUPABASE_PUBLISHABLE_KEY=your_publishable_key
SUPABASE_SECRET_KEY=your_server_only_secret_key

SUPABASE_JWT_ISSUER=https://YOUR_PROJECT_ID.supabase.co/auth/v1
SUPABASE_JWKS_URL=https://YOUR_PROJECT_ID.supabase.co/auth/v1/.well-known/jwks.json

API_URL=http://127.0.0.1:8000
ALLOWED_ORIGINS=http://localhost:8501
```

Never commit `.env` to source control.

Never expose `SUPABASE_SECRET_KEY` or any server-only secret in frontend code.

---

## Running the Application

The Streamlit app talks to the FastAPI backend, so start the backend first
(see [Running the FastAPI Backend](#running-the-fastapi-backend)).

### Streamlit application

Start the Streamlit application in a second terminal:

```powershell
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

Log in, or create a patient account. Clinician, reviewer, and admin roles are
assigned in Supabase and see the clinician dashboard instead of the patient page.

### Patient page

The patient can:

1. Start a new assessment.
2. Describe symptoms in free text.
3. Answer follow-up questions while the progress indicator updates.
4. Review the urgency result and the reasons behind it.
5. View the suggested care pathway.
6. Download a summary for a doctor.
7. Note the anonymous patient code.
8. Delete stored data.

### Clinician dashboard

The clinician dashboard shows urgency counts and a session table. After a
session is selected, it is organized into tabs:

| Tab | Contents |
|---|---|
| Overview | Structured summary, symptom timeline, continuity note, PDF download |
| Explainability | Confidence, reasons, clinician-review flag, session state |
| Trace | Decision trace and LLM latency |
| Workflow | LangGraph diagram with the executed path highlighted |
| Feedback | Clinician assessment form and patient data deletion |

In the workflow diagram, green nodes were executed, the red node marks emergency
escalation, and dashed grey nodes were not reached.

The optional dashboard PIN is only a local demonstration feature. It is not a replacement for proper authentication.

---

## Running the FastAPI Backend

The FastAPI backend uses Supabase authentication and persistence.

Start the backend:

```powershell
uvicorn api.main:app --reload
```

For a specific host and port:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Available documentation:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc
```

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

The backend supports:

- JWT authentication
- Patient assessment creation
- Patient messages
- Patient assessment retrieval
- Patient data deletion
- Clinician session listing
- Clinician session details (including the workflow diagram)
- Clinician summaries
- PDF report downloads
- Clinician feedback
- Rate limiting
- Audit logging

---

## Supabase Setup

1. Create a Supabase project.
2. Open the Supabase SQL editor.
3. Run:

```text
supabase/migrations/001_initial_schema.sql
```

The migration creates:

- User profiles
- Patient records
- Assessment sessions
- Messages
- Symptom events
- Audit logs
- Clinician feedback
- Role-based helper functions
- Row-level security policies
- Indexes
- User profile creation trigger

### Supported roles

| Role | Description |
|---|---|
| `patient` | Can create and view their own assessments |
| `clinician` | Can review patient sessions |
| `reviewer` | Can review patient sessions |
| `admin` | Can manage administrative data and roles |

### Supabase authentication

The API expects a valid Supabase access token:

```http
Authorization: Bearer <access-token>
```

The API validates:

- Token signature
- Expiration
- Issuer
- Audience
- Subject
- Authentication role
- Supabase profile role

---

## Command-Line Tools

### Deterministic fallback chat

This mode does not require an LLM:

```powershell
python scripts/cli_fallback.py
```

Example input:

```text
I have had fever for three days
```

### Agent without an LLM

```powershell
python scripts/chat_cli.py --no-llm --trace
```

The `--trace` option displays tool calls and workflow information.

### Agent with an LLM

```powershell
python scripts/chat_cli.py --trace
```

### Engine demonstration

```powershell
python scripts/demo_engine.py
```

### Memory demonstration

```powershell
python scripts/demo_memory.py
```

This uses a temporary demonstration database and shows:

- Anonymous patient code generation
- Session persistence
- Continuity notes
- Symptom timelines

### LLM connectivity check

```powershell
python scripts/test_llm.py
```

### Extractor test

```powershell
python scripts/test_extractor.py
```

### Adversarial test

```powershell
python scripts/test_adversarial.py
```

The adversarial test checks cases such as:

- Prompt injection
- Requests for diagnosis
- Requests for medication
- Missing information
- Hinglish input
- Child-related scenarios
- Unsafe or ambiguous wording

---

## Testing

Run the complete test suite:

```powershell
python -m pytest -q
```

The test suite covers:

- API security
- Authentication and authorization
- Red-flag detection
- Triage rules
- Condition evaluation
- Answer parsing
- Knowledge-base validation
- Memory and timelines
- Agent orchestration
- Extractor behavior
- Output guardrails
- Safety review
- Clinician summaries
- Streamlit application behavior

Run a specific test file:

```powershell
python -m pytest tests/test_red_flags.py -q
```

Run tests matching a keyword:

```powershell
python -m pytest -k "red_flag" -q
```

Validate the knowledge base:

```powershell
python scripts/check_knowledge.py
```

---

## Evaluation

### Development vignette evaluation

Run the development vignettes:

```powershell
python eval/run_vignettes.py
```

These vignettes are useful as smoke tests, but they are not independent validation because they were created alongside the rules.

### Offline held-out evaluation

Run the rule-based evaluation:

```powershell
python eval/run_eval.py
```

The command evaluates:

- Main rule-based system
- Component ablations
- Baseline approaches
- Per-group performance
- Failure cases

Results are written under:

```text
eval/results/
```

### Live evaluation

Run the live extraction and review evaluation:

```powershell
python eval/run_eval.py --live --extraction --tag live1
```

For a smaller test:

```powershell
python eval/run_eval.py --live --extraction --limit 10 --tag smoke
```

The live evaluation requires a working LLM configuration and may incur provider rate limits.

### Clinician review sheet

Generate or use the review sheet:

```text
eval/review_sheet.csv
```

After a qualified reviewer completes the sheet:

```powershell
python eval/run_eval.py `
  --reviewed eval/review_sheet_filled.csv `
  --tag clinician
```

Do not alter held-out cases after their checksum has been recorded.

---

## Knowledge Base

The knowledge base is stored as JSON and CSV files.

### Protocols

Complaint-specific protocols are stored in:

```text
knowledge/protocols/
```

Protocols include:

- Abdominal pain
- Chest pain
- Cough and breathing problems
- Dizziness and weakness
- Fever
- Gastrointestinal upset
- Headache
- Minor injury
- Other complaints
- Sore throat and ear symptoms
- Urinary symptoms

### Shared knowledge files

```text
knowledge/common.json
knowledge/modifiers.json
knowledge/pathways.json
knowledge/red_flags.json
knowledge/synonyms.json
knowledge/symptom_vocab.csv
```

### Vocabulary generation

The vocabulary can be rebuilt with:

```powershell
python data/prepare_vocab.py
```

Optional raw data files are:

```text
data/raw/Symptom-severity.csv
data/raw/dataset.csv
```

The vocabulary-generation script also writes:

```text
data/dataset_report.txt
```

Always validate changes:

```powershell
python scripts/check_knowledge.py
```

---

## Safety Design

TriageGuide uses multiple safety layers.

### 1. Deterministic red-flag gate

Patient text is checked for emergency indicators before the system makes an LLM call.

This protects the system from missing urgent information when:

- The model is unavailable
- The model returns invalid JSON
- The model misunderstands the message
- The network is unavailable

### 2. Fact validation

The LLM proposes structured facts, but Python validates:

- Data types
- Allowed values
- Numeric ranges
- Quoted denials
- Known field names
- Required structure

Invalid or unsupported facts are rejected.

### 3. Deterministic urgency calculation

The urgency engine uses the validated patient state and knowledge-base rules.

The model does not directly assign the final urgency.

### 4. Escalation-only safety review

The optional AI safety review can raise urgency, but it cannot lower an existing urgency level.

It also cannot raise a normal case directly to emergency unless the deterministic emergency gate supports it.

### 5. Output guard

Generated questions and review text are checked for:

- Diagnosis wording
- Medication recommendations
- Unsafe treatment claims
- Prompt-injection output
- Unsupported medical conclusions

Unsafe generated content is replaced with fixed fallback text.

### 6. Refusal handling

Requests such as:

```text
What disease do I have?
Which tablet should I take?
```

receive fixed refusal responses instead of unconstrained model output.

### 7. LLM failure fallback

When an LLM call fails, the system continues with deterministic behavior where possible.

After repeated failures, the circuit breaker temporarily stops additional model calls.

### 8. Clinician summary safety

Clinician summaries are generated from validated state and stored records rather than unrestricted model-generated text.

Patient-provided content is escaped before it is included in summaries.

---

## Data Storage and Privacy

### Local mode

Local demonstrations and command-line tools use SQLite.

By default, data is stored in:

```text
triageguide.db
```

The location can be changed with:

```dotenv
TRIAGE_DB=C:\path\to\triageguide.db
```

Local databases should not be committed to Git.

### Supabase mode

The API uses Supabase for:

- Authentication
- Patient profiles
- Sessions
- Messages
- Symptom events
- Audit logs
- Clinician feedback

### Privacy characteristics

The application uses anonymous patient codes for continuity. It is designed not to require a patient's name or phone number.

However, this project is still a prototype. Do not enter real patient information.

A production deployment would require:

- Formal privacy review
- Data retention rules
- Encryption
- Secure backups
- Access monitoring
- Audit review
- Incident response
- Regulatory assessment
- Clinical governance

---

## API Overview

### Health

```http
GET /health
```

### Current user

```http
GET /auth/me
Authorization: Bearer <access-token>
```

### Create a patient assessment

```http
POST /patient/assessments
Authorization: Bearer <patient-token>
```

### Submit a patient message

```http
POST /patient/assessments/{session_id}/messages
Authorization: Bearer <patient-token>
Content-Type: application/json
```

Request body:

```json
{
  "text": "I have had a fever for three days"
}
```

### Retrieve an assessment

```http
GET /patient/assessments/{session_id}
Authorization: Bearer <patient-token>
```

### Retrieve assessment messages

```http
GET /patient/assessments/{session_id}/messages
Authorization: Bearer <patient-token>
```

### Download a patient summary

```http
GET /patient/assessments/{session_id}/download
Authorization: Bearer <patient-token>
```

### Delete patient data

```http
DELETE /patient/me
Authorization: Bearer <patient-token>
```

### List clinician sessions

```http
GET /clinician/sessions
Authorization: Bearer <clinician-token>
```

### Retrieve clinician session details

```http
GET /clinician/sessions/{session_id}
Authorization: Bearer <clinician-token>
```

The response includes `summary`, `state`, `report`, `workflow`, `timeline`,
`continuity`, and `feedback`.

### Retrieve clinician summary

```http
GET /clinician/sessions/{session_id}/summary
Authorization: Bearer <clinician-token>
```

### Download a clinician PDF

```http
GET /clinician/sessions/{session_id}/pdf
Authorization: Bearer <clinician-token>
```

### Submit clinician feedback

```http
POST /clinician/sessions/{session_id}/feedback
Authorization: Bearer <clinician-token>
Content-Type: application/json
```

Example request:

```json
{
  "outcome": "incorrect_urgency",
  "comment": "The patient should be reviewed sooner."
}
```

### Delete a patient as a clinician

```http
DELETE /clinician/patients/{patient_code}
Authorization: Bearer <clinician-token>
```

---

## Troubleshooting

### Python is not recognized

Install Python 3.11 or newer and enable the option to add Python to PATH.

Verify:

```powershell
python --version
```

### Virtual environment will not activate

Run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### LLM model not found

Check the current provider model list and update:

```dotenv
GROQ_MODEL=your_current_model_id
```

### LLM requests fail

Check:

- `GROQ_API_KEY`
- Internet connectivity
- Provider availability
- Model name
- Network or VPN restrictions
- Provider rate limits

The deterministic mode can still be used:

```powershell
python scripts/chat_cli.py --no-llm --trace
```

### API fails during startup

The FastAPI API requires all Supabase authentication variables:

```dotenv
SUPABASE_URL
SUPABASE_SECRET_KEY
SUPABASE_JWT_ISSUER
SUPABASE_JWKS_URL
```

Confirm that the Supabase migration has been applied.

### Streamlit cannot connect to the API

Confirm that the API is running:

```powershell
uvicorn api.main:app --reload
```

Then verify:

```powershell
curl http://127.0.0.1:8000/health
```

Check that `API_URL` matches the backend address.

### Session expired message

The Streamlit app logs you out when the API returns `401`. Log in again to get a
fresh token.

### Workflow diagram is blank or shows an error

The diagram is rendered in the browser with Mermaid loaded from a CDN, so the
browser needs internet access. Open the browser console for the exact error. If
the diagram renders but nothing is highlighted, confirm that
`GET /clinician/sessions/{session_id}` builds the diagram with
`workflow_diagram(report)` and that `report` contains `tool_log`.

### Streamlit warns about `use_container_width` or `width="stretch"`

These parameters changed between Streamlit versions. Use whichever style your
installed version supports, and use it consistently across `app.py`.

### Knowledge validation fails

Run:

```powershell
python scripts/check_knowledge.py
```

Review the reported JSON file, field name, or protocol rule.

---

## Limitations

- This system is not clinically validated.
- It must not be used to make real medical decisions.
- The system does not diagnose conditions.
- The system does not prescribe medication.
- The knowledge base is educational and may contain incomplete rules.
- Development vignettes are not independent validation data.
- Held-out labels may not have been verified by clinicians.
- LLM output can be incorrect or unavailable.
- Provider models and APIs may change.
- The optional Gemini provider is not fully tested.
- The dashboard PIN is not real authentication.
- The patient progress indicator is an estimate based on the number of answers, not the real workflow stage.
- Logging out clears the local Streamlit session only; the Supabase token stays valid until it expires.
- Local SQLite storage is not suitable for production deployment.
- Closing a browser session may interrupt an unfinished live conversation.
- Emergency numbers and care guidance must be verified before deployment in any region.

---

## Development Guidelines

Before opening a pull request:

1. Run the test suite:

   ```powershell
   python -m pytest -q
   ```

2. Validate the knowledge base:

   ```powershell
   python scripts/check_knowledge.py
   ```

3. Keep safety-critical decisions deterministic.
4. Add focused tests for new behavior.
5. Do not commit `.env`, API keys, database files, or credentials.
6. Update this README when commands or configuration change.
7. Review changes to `knowledge/` carefully.
8. Document the source of any clinical rule added to a protocol.
9. Run the adversarial checks when changing prompts or guardrails.
10. Do not claim clinical accuracy based only on development-vignette agreement.
11. When adding or renaming a LangGraph node, update `TOOL_TO_NODE` and the diagram in `agent/workflow_visualization.py`.

---

## License and Attribution

TriageGuide is intended for educational and research use.

Review the repository license and the licenses of all third-party dependencies before redistribution.

The project uses public datasets and external model providers where configured. Follow the terms of each dataset, API, and dependency.