"""Central configuration for TriageGuide.

Purpose: one place that knows where files live and reads settings from the
.env file, so no other module hard-codes paths, keys or model names.
"""
import os
from pathlib import Path

try:  # python-dotenv is in requirements.txt, but do not crash tests without it
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent
if load_dotenv:
    load_dotenv(BASE_DIR / ".env")

KNOWLEDGE_DIR = BASE_DIR / "knowledge"
PROTOCOL_DIR = KNOWLEDGE_DIR / "protocols"
DATA_DIR = BASE_DIR / "data"
EVAL_DIR = BASE_DIR / "eval"
DB_PATH = Path(os.getenv("TRIAGE_DB", str(BASE_DIR / "triageguide.db")))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

EMERGENCY_NUMBER = os.getenv("EMERGENCY_NUMBER", "112")
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "15"))
# A model call slower than this is abandoned and the agent uses its deterministic fallback for that step.
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "12"))

# Optional PIN for the clinician dashboard in the web app. Empty = no PIN (fine for a local demo only).
DASHBOARD_PIN = os.getenv("DASHBOARD_PIN", "")
