"""Resume-JD Match Checker: Streamlit app.

Run with:  streamlit run app.py
"""

import os

import streamlit as st

from llm import get_feedback, is_busy_error
from matcher import load_embedder, match, read_pdf

st.set_page_config(page_title="Resume–JD Match Checker", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading the embedding model (first run only)...")
def get_embedder():
    return load_embedder()


def get_api_key() -> str:
    """Look for the Gemini key in Streamlit secrets, then in env vars."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GEMINI_API_KEY", "")


st.title("📄 Resume–JD Match Checker")
st.write(
    "Upload your resume and paste a job description. "
    "The app gives a match score and shows which skills are missing."
)

left, right = st.columns(2)
with left:
    resume_file = st.file_uploader("Your resume (PDF)", type=["pdf"])
with right:
    jd_text = st.text_area("Job description", height=220, placeholder="Paste the JD here...")

use_llm = st.checkbox("Also get written feedback from Gemini (needs an API key)", value=bool(get_api_key()))

if st.button("Check match", type="primary"):
    if resume_file is None or not jd_text.strip():
        st.warning("Please upload a resume and paste a job description.")
        st.stop()

    resume_text = read_pdf(resume_file)
    if len(resume_text.strip()) < 50:
        st.error("Could not read text from this PDF. Is it a scanned image? Try a text-based PDF.")
        st.stop()

    with st.spinner("Comparing..."):
        result = match(resume_text, jd_text, get_embedder())

    # ---- scores ----
    st.subheader("Scores")
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall match", f"{result.final_score:.0f} / 100")
    c2.metric("Meaning match", f"{result.semantic_score:.0f} / 100")
    c3.metric("Skill match", f"{result.skill_score:.0f} / 100")
    st.progress(min(result.final_score / 100, 1.0))
    st.caption("Overall = 50% meaning match + 50% skill match. Treat it as a guide, not a verdict.")

    # ---- skills ----
    st.subheader("Skills")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("**✅ In JD and in resume**")
        st.write(", ".join(result.matched_skills) or "—")
    with s2:
        st.markdown("**❌ In JD, missing from resume**")
        st.write(", ".join(result.missing_skills) or "Nothing missing 🎉")
    with s3:
        st.markdown("**➕ In resume, not asked in JD**")
        st.write(", ".join(result.extra_skills) or "—")

    # ---- weak lines ----
    with st.expander("JD lines your resume covers least"):
        for line, sim in result.weak_jd_lines:
            st.write(f"- {line}  _(similarity {sim:.2f})_")

    # ---- LLM feedback ----
    if use_llm:
        key = get_api_key()
        if not key:
            st.info("No GEMINI_API_KEY found, so written feedback is skipped. See the README to add one.")
        else:
            st.subheader("Written feedback (Gemini)")
            try:
                with st.spinner("Asking Gemini..."):
                    st.markdown(get_feedback(resume_text, jd_text, key))
            except Exception as e:  # network, quota, bad key...
                if is_busy_error(str(e)):
                    st.warning(
                        "Gemini is busy right now (too many requests on Google's side). "
                        "Your scores above are still valid. Please try again in a few minutes."
                    )
                else:
                    st.error(f"Gemini call failed: {e}")
