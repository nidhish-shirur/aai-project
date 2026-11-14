# TriageGuide - Phases 0 to 7 (setup, knowledge base, LLM extractor, memory, safety core, agent, guardrails, web app)

Educational decision-support prototype. **Not a medical device. It never diagnoses.**
Everything in this package runs on Windows with free tools. The safety core (Phase 4)
needs **no API key and no internet**.

## Step-by-step on Windows with VS Code

### A. One-time installs
1. Install **Python 3.11 or newer** from python.org. On the first installer screen tick **"Add python.exe to PATH"**.
2. Install **VS Code** (code.visualstudio.com), then open it and install the **Python** extension (Microsoft) from the Extensions tab (Ctrl+Shift+X).
3. (Optional) Install **Git for Windows**.

### B. Open the project
4. Unzip `triageguide.zip` somewhere simple, for example `C:\Projects\triageguide`.
5. In VS Code: **File > Open Folder** and choose the `triageguide` folder (the one that contains `config.py`).
6. Open a terminal: **Terminal > New Terminal** (PowerShell).

### C. Create the environment (Phase 0)
7. Create a virtual environment:
   ```powershell
   python -m venv .venv
   ```
8. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell says scripts are disabled, run once
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and activate again
   (or use the "Command Prompt" terminal and run `.venv\Scripts\activate.bat`).
   You should see `(.venv)` at the start of the prompt.
9. Tell VS Code to use it: press **Ctrl+Shift+P**, type **Python: Select Interpreter**, pick the one inside `.venv`.
10. Install the packages:
    ```powershell
    pip install -r requirements.txt
    ```

### D. Connect the LLM (Phase 0 test)
11. Get a free key at https://console.groq.com/keys .
12. Create your settings file and open it:
    ```powershell
    copy .env.example .env
    code .env
    ```
    Paste the key after `GROQ_API_KEY=` and save. Never share or commit `.env`.
13. Test the connection:
    ```powershell
    python scripts/test_llm.py
    ```
    Expected: a short JSON reply and "Phase 0 complete". If it says the **model was not found**,
    open https://console.groq.com/docs/models, copy a current model ID into `GROQ_MODEL` in `.env`, and retry.
    (Phases 1 and 4 below do not need this step.)

### E. Knowledge base (Phase 1)
14. Validate every protocol and rule:
    ```powershell
    python scripts/check_knowledge.py
    ```
    Expected: `Knowledge base OK.`
15. Build the symptom vocabulary:
    ```powershell
    python data/prepare_vocab.py
    ```
    For the full vocabulary, download the Kaggle files (free account) from
    https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset ,
    create the folder `data\raw` and put `Symptom-severity.csv` and `dataset.csv` in it, then run the command again.
    It also writes `data/dataset_report.txt` (row count and duplicate count of the Kaggle data).

### F. Safety core (Phase 4)
16. Run all tests (VS Code also shows them under the beaker icon):
    ```powershell
    python -m pytest -q
    ```
    Expected: all tests pass (411 at the time of writing).
17. Run the vignette evaluation table:
    ```powershell
    python eval/run_vignettes.py
    ```
18. Try the engine by hand (no LLM needed):
    ```powershell
    python scripts/cli_fallback.py
    ```
    Type for example `my father has chest pain`, or `I have had a fever for three days`, and answer the questions.
    Also try `python scripts/demo_engine.py`.

### G. LLM extractor (Phase 2)
19. Make sure `.env` has a working model (`GROQ_MODEL=openai/gpt-oss-120b` worked in testing).
20. Run the live extractor check (needs your key):
    ```powershell
    python scripts/test_extractor.py
    ```
    It sends 8 sample messages (English, Hinglish, short answers like "not really") to the model and prints,
    for each, what was **applied**, **rejected** and the time taken. A line starting with `!! LLM FAILED` means the
    model call failed and the deterministic fallback was used. **Paste the whole output back to me** so I can tune the prompt.
    The unit tests for this module use a scripted mock and need no key (`python -m pytest -q`).

### H. Memory and timeline (Phase 3)
21. Run the demo (uses a throw-away database, so nothing is saved to your real one):
    ```powershell
    python scripts/demo_memory.py
    ```
    Expected: a continuity note ("7 day(s) ago: fever, assessed as moderate") and a two-line symptom timeline.
22. Try it for real with the fallback chat. Finish one visit, **write down the patient code** it prints (like `P-3AC59E`),
    then run `python scripts/cli_fallback.py` again and enter that code at the first prompt. It will remember the earlier visit.
    Sessions are saved in `triageguide.db` in the project folder (ignored by Git). No name or phone number is ever stored.

### I. The agent: chat with tool calling (Phase 5)
23. Without an API key (deterministic mode, still goes through the agent):
    ```powershell
    python scripts/chat_cli.py --no-llm --trace
    ```
24. With the LLM (needs your Groq key in `.env`):
    ```powershell
    python scripts/chat_cli.py --trace
    ```
    Try: `I am a 30 year old man, fever for 3 days and a bad headache`. The agent should NOT ask your age,
    sex or duration again. `--trace` prints every tool the agent used each turn (red_flag_gate, extract_facts,
    plan_next, ask_question, compute_urgency) and the seconds per turn. **Paste a full conversation back to me**
    (including the `[tool]` lines and `[time]`), so I can tune the planner prompt and check latency.
    Also try: `my father has crushing chest pain` (emergency, zero LLM calls) and a second visit with your
    patient code (the agent asks whether the earlier symptoms have gone away).

### J. Guardrails (Phase 6)
25. Try the refusals in the chat: `python scripts/chat_cli.py --trace`, then type
    `I have a fever for 2 days. what disease do I have?` The agent must answer with fixed text
    ("I can't tell you what condition this might be...") and carry on asking triage questions.
    `which tablet should I take?` gets the medicine refusal.
26. Run the live adversarial check (needs your key):
    ```powershell
    python scripts/test_adversarial.py
    ```
    12 hostile or awkward messages (prompt injection, requests to diagnose, Hinglish, a child). Every line should say PASS.
    If it says `Cannot reach the LLM` or a case says `LLM was unreachable`, the connection failed (a `403 Access denied` from Groq usually
    means a VPN, college network or region block: try a phone hotspot) and that case is not a valid test.
    **Paste the whole output back to me.**

### K. Latency log
27. Run a normal conversation with the trace on: `python scripts/chat_cli.py --trace`.
    Every turn now prints one `[llm]` line per model call (`extract`, `plan`, `review`) with its seconds, number of attempts,
    `RATE-LIMITED` waits and output tokens, and a `[time]` line that splits the turn into time in the LLM and everything else.
    At the end a **Latency summary** gives totals per purpose. The same numbers are stored in the report (`latency`).
    **Paste one full conversation back to me.** It shows whether slow turns come from Groq's rate limits (look for `RATE-LIMITED`),
    long model outputs (high token counts), or a slow connection (long seconds with 1 attempt).

    **Speed-ups from the first latency log:** (1) a bare answer such as `no`, `yes`, `101 F` or `none` to the question just asked is now
    read by Python alone (trace shows `fast_path: True`, no `[llm] extract` line); anything longer still goes to the LLM, because it
    may carry extra facts. (2) A model call slower than `LLM_TIMEOUT_S` (default 12 s, set in `.env`) is abandoned and the fixed fallback
    is used for that step. (3) The Groq SDK's silent retries are switched off so the log shows every wait; the client does one quick retry for
    a network blip or busy server (`transient_retries`) and handles rate limits itself.

### L. Web app and clinician summary (Phase 7)
29. Install the new package (once): `pip install -r requirements.txt` (adds `streamlit`).
30. Start the app:
    ```powershell
    streamlit run app.py
    ```
    A browser tab opens at http://localhost:8501.
    * **Patient** page: tick or untick "Use the AI model" (unticked = fixed question order, no key or internet needed), press *Start assessment*
      and chat. At the end you get the urgency card, a patient code, a **Download summary for your doctor** button, a
      "How did the assistant decide?" panel (the tool calls) and *Delete my data*.
    * **Clinician dashboard** page (left sidebar): urgency counts, a table sorted most-urgent first, the one-page summary of any session
      (what the patient reports / denies / could not answer, why this urgency, red flags, AI second opinion, timeline, the patient's own words),
      the decision trace, LLM latency, and delete. Unfinished sessions are hidden unless you tick the box.
    * Optional: put `DASHBOARD_PIN=1234` in `.env` to require a PIN for the dashboard (a demo-grade lock, not real authentication).
31. Take 2 screenshots (patient result and dashboard) for your report and demo slides.

### M. Evaluation (Phase 8)
32. Offline run (no key, about 15 seconds): the rule-based system on 48 held-out cases plus five ablations.
    ```powershell
    python eval/run_eval.py
    ```
    Read `eval/results/RESULTS_latest.md`. `eval/results/FROZEN_v1_RESULTS.md` is the first run, recorded before any tuning.
33. With your Groq key (adds the LLM-only baseline and the rules + AI review; about 100 calls, a few minutes):
    ```powershell
    python eval/run_eval.py --live --extraction --tag live1
    ```
    Optional, slow: add `--full` (complete agent with LLM on every case; use `--limit 10` first).
34. Best practice: give `eval/review_sheet.csv` to a doctor or nurse to fill `clinician_label`, then
    `python eval/run_eval.py --reviewed eval/review_sheet_filled.csv --tag clinician`.
    The program refuses to run if `heldout_cases.json` was edited after its checksum was recorded.
    Do not edit labels after seeing results.

### N. Save your work (optional)
35. ```powershell
    git init
    git add .
    git commit -m "Phases 0 to 8"
    ```
    `.gitignore` already keeps `.env`, `.venv` and the database out of Git.

## What each file is for

| File | Purpose |
|---|---|
| `agent/orchestrator.py` | **The agent**: per turn it runs tools (red-flag gate, extractor, planner, rules engine). The LLM plans questions; Python validates the plan and decides urgency |
| `app.py` | **Streamlit web app**: patient chat + clinician dashboard (PIN optional) |
| `agent/summary.py` | **Clinician summary**: built by plain Python from the validated record (no LLM can add anything); text is escaped so patient words cannot inject Markdown or links |
| `agent/guard.py` | **Output guard**: rejects LLM-written text with diagnosis or medicine language; detects patient requests for a diagnosis or medicine |
| `agent/safety_review.py` | **Escalate-only LLM second opinion**: may raise urgency (capped at *urgent*), never lower it; its sentence is guarded |
| `agent/llm_client.py` | Provider-independent LLM wrapper (Groq; optional untested Gemini; `MockLLM` for tests) |
| `agent/extractor.py` | **LLM extractor**: free text to facts. The LLM proposes, Python validates (types, ranges, quoted denials) with a no-LLM fallback |
| `memory/db.py` | **SQLite memory**: sessions, messages, reports, anonymous patient codes, continuity note, delete/purge for privacy |
| `memory/timeline.py` | Builds the symptom timeline ("Day 1: Fever, Day 4: Fever + Cough") |
| `config.py` | Paths and settings from `.env` (keys, model names, emergency number) |
| `agent/state.py` | `PatientState`: everything known about the patient this session (True / False / unknown) |
| `engine/conditions.py` | Small safe condition language used inside the JSON rules (no `eval`) |
| `engine/protocols.py` | Loads protocols, decides what is still worth asking, selects a protocol from keywords |
| `engine/red_flags.py` | **Deterministic emergency gate** (words + facts, handles "no chest pain") |
| `engine/triage_rules.py` | **Urgency engine**: red flags, rules, risk modifier, cautious floor, escalate-only merge |
| `engine/escalation.py` | When a human clinician must review (missing data, conflicts, children, pregnancy...) |
| `engine/pathways.py` | Urgency level to care pathway and all user-facing text (from JSON) |
| `engine/answer_parser.py` | Turns "yes", "101 F", "2 weeks", "sugar and bp" into typed values |
| `engine/vocab.py` | Maps free text ("pet dard") to canonical symptoms (`abdominal_pain`) |
| `knowledge/protocols/*.json` | 11 complaint protocols: fields, red flags, rules, warning signs |
| `knowledge/red_flags.json` | Global red flags and the list of safety facts |
| `knowledge/common.json`, `modifiers.json`, `pathways.json`, `synonyms.json` | Shared fields, risk modifiers, wording, synonyms |
| `data/prepare_vocab.py` | Builds `knowledge/symptom_vocab.csv` (uses Kaggle for symptom names only) |
| `data/gold_vignettes.json` | 34 development vignettes (9 emergency, 9 urgent, 8 moderate, 8 low) |
| `eval/vignette_runner.py`, `run_vignettes.py` | Dev-set smoke test (circular: written by the same author as the rules) |
| `eval/heldout_cases.json` (+ `.sha256`) | 48 held-out cases with frozen, AI-drafted labels (not clinician-verified) |
| `eval/run_eval.py` | Evaluation runner: main comparison, ablations, per-group table, failure analysis |
| `eval/metrics.py`, `sim_patient.py`, `ablations.py`, `baselines.py` | Metrics, simulated patient, component switch-offs, LLM-only baseline |
| `eval/extraction_cases.json`, `extraction_eval.py` | 30 annotated messages for extractor accuracy |
| `eval/latency_report.py`, `export_review_sheet.py` | Latency stats from saved sessions; blind sheet for a clinician |
| `scripts/test_adversarial.py` | Live check: 12 hostile messages against the real LLM |
| `scripts/chat_cli.py` | Terminal chat with the agent (`--trace` shows tool calls) |
| `scripts/*.py` | LLM test, live extractor test, knowledge checker, fallback chat, demos |
| `tests/` | pytest suite (red flags, rules, parser, knowledge, vignettes) |

## Important honesty notes
* **Timeline:** a returning patient is asked whether earlier symptoms have gone away. "Yes" marks them resolved **on the date they told us**,
  not the (unknown) real date, so the timeline may show an old symptom as active a little longer than it really was.
  Symptoms other than the main complaint are dated when first reported, not when they began.
* **Phase 7 notes:** the clinician summary is deterministic on purpose: a doctor must be able to trust that every line comes from the record.
  The dashboard PIN is a demo lock, not authentication; real use would need accounts, HTTPS and an audit log. The patient's typed words are
  shown as plain text in the chat and escaped in the summary (a typed `![x](http://...)` would otherwise make the clinician's browser contact
  that address). The app has headless tests (Streamlit's `AppTest`): they use a temporary database and never call an LLM.
  A returning patient can type their code on the start page; closing the browser tab mid-chat loses the live conversation (state up to the last answer is saved).
* **Phase 6 safety design:** the LLM writes only question wording and one review sentence; both pass `agent/guard.py` or are replaced by
  fixed JSON wording. Refusals of "what disease do I have?" / "which medicine?" are fixed text, never LLM output. The second-opinion
  review can only raise urgency, never to *emergency* (that stays with the deterministic gate), and every raise adds a clinician-review flag.
  The guard is regex, so it is conservative and can miss paraphrases; that is why the safety-critical text is fixed, not generated.
  The unit tests include mutation checks done by hand (disabling each protection makes a test fail). Prompt-injection tests are
  scripted, so run `scripts/test_adversarial.py` for the real model.
  After 3 failed LLM calls in a row the agent stops calling it (circuit breaker) and continues in deterministic mode until a call works again.
  The review adds one more LLM call at the end of a conversation (about 1 to 2 s).
* **Phase 5 safety design:** the emergency gate runs on the patient's words before any LLM call; the LLM's plan is validated
  (it cannot stop while required questions remain, invent fields, ask optional questions first, or use diagnosis/medicine
  wording in a question); a failing LLM falls back to the fixed question order; urgency is always decided by the rule engine.
  Each turn can use up to two LLM calls (extract + plan), so latency is not yet measured against the 4 s target: the live run will tell.
  An agent object lives in memory; a closed app cannot yet resume a half-finished conversation (state is saved, resuming is not built).
* **Phase 2 safety rule:** the extractor stores the raw message before calling the LLM, so the red-flag gate still works if the LLM fails.
  A denial ("no") is only accepted if the model quotes the patient's own words; a "yes" is always kept. The complaint is set once and not
  switched later (a limitation: one protocol per session; Phase 5 will handle a second complaint).
* The Gemini client is written from the public API format and is **untested**.
* **The rules are not clinically validated.** They are student-written from general public guidance.
  Have a doctor or faculty reviewer check `knowledge/` and record who reviewed it and when.
  Add the source URL/clause to each protocol's `source_notes` as you verify it.
* **The 34 vignettes were written together with the rules, so 100% agreement is a smoke test, not evidence of accuracy.**
  For the final evaluation, write a *separate* held-out set (ideally labelled by someone who has not seen the rules)
  and report that result.
* Verify the helpline numbers in `knowledge/pathways.json` (112 emergency, Tele-MANAS 14416) before any demo.
* `MAX_QUESTIONS` default is 15: fallback mode asks one field per question and some protocols have 11 required fields.
  In LLM mode (Phase 2/5) one answer can fill several fields, so the plan's "about 4 to 7 questions" target stays realistic.

## Next phases
Phase 9 (documentation, viva Q&A, demo script).
