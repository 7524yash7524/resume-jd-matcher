"""Optional LLM step: ask Gemini for plain-language feedback.

The app works without this. If no API key is set, the app just shows
the score and skill lists.
"""

from __future__ import annotations

import os
import time

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Tried in order if a model is not available (Google retires old ones).
FALLBACK_MODELS = ["gemini-flash-latest"]

PROMPT = """You are helping a student check a resume against a job description.

Rules:
- Use only the text given below. Do not invent experience the resume does not show.
- Use simple, plain English.
- Keep the whole answer under 250 words.

Give exactly three sections in Markdown:
### Missing skills
Skills or tools the JD asks for that the resume does not show (bullet list).
### Strong points
Things in the resume that match the JD well (bullet list, max 4).
### Quick fixes
3 short, honest suggestions to improve the resume for this JD.

JOB DESCRIPTION:
{jd}

RESUME:
{resume}
"""


def build_prompt(resume_text: str, jd_text: str, max_chars: int = 12000) -> str:
    return PROMPT.format(jd=jd_text[:max_chars], resume=resume_text[:max_chars])


def get_feedback(resume_text: str, jd_text: str, api_key: str, model: str = DEFAULT_MODEL) -> str:
    """Call Gemini and return its Markdown answer.

    If the chosen model is not found (404), try the fallback models.
    """
    from google import genai

    client = genai.Client(api_key=api_key)
    prompt = build_prompt(resume_text, jd_text)

    last_error = None
    for name in [model] + [m for m in FALLBACK_MODELS if m != model]:
        for attempt in range(2):  # one retry per model when the server is busy
            try:
                response = client.models.generate_content(model=name, contents=prompt)
                return response.text or "The model returned an empty answer."
            except Exception as e:
                msg = str(e)
                last_error = e
                if is_busy_error(msg):
                    if attempt == 0:
                        time.sleep(3)
                        continue  # retry the same model once
                    break  # still busy: try the next model
                if "NOT_FOUND" in msg or "404" in msg:
                    break  # model missing: try the next model
                raise
    raise last_error


def is_busy_error(message: str) -> bool:
    """True when Google says the model is overloaded or rate-limited."""
    return any(code in message for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"))
