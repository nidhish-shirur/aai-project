"""Phase 0 check: can Python reach the Groq API with your key?

Run from the project root:   python scripts/test_llm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


def main() -> int:
    if not config.GROQ_API_KEY or "paste_your" in config.GROQ_API_KEY:
        print("GROQ_API_KEY is missing. Copy .env.example to .env and paste your key.")
        return 1

    try:
        from groq import Groq
    except ImportError:
        print("The 'groq' package is not installed. Run: pip install -r requirements.txt")
        return 1

    client = Groq(api_key=config.GROQ_API_KEY)
    print(f"Calling Groq model: {config.GROQ_MODEL}")
    try:
        reply = client.chat.completions.create(
            model=config.GROQ_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": "Reply with one short JSON object only."},
                {"role": "user", "content": 'Return {"status": "ok", "message": "hello from the LLM"}'},
            ],
        )
    except Exception as exc:  # network, auth or model-name problems
        print("API call failed:", exc)
        print("If the error mentions the model, pick a current model ID at "
              "https://console.groq.com/docs/models and set GROQ_MODEL in .env")
        return 1

    print("Reply:", reply.choices[0].message.content)
    print("\nPhase 0 complete: the LLM connection works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
