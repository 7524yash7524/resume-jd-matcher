# Resume–JD Match Checker

https://resume-jd-matcher-elcdj8mtm7vyvqf5y3rxx5.streamlit.app/


A small Streamlit app that checks how well a resume fits a job description (JD).

You upload a resume (PDF) and paste a JD. The app gives you:

- an **overall match score** out of 100
- the **skills the JD asks for that your resume is missing**
- the **JD lines your resume covers least**
- optional **written feedback from an LLM** (Google Gemini)

**Live demo:** _add your Streamlit Cloud link here_

![screenshot](screenshot.png) <!-- add a screenshot after you run it -->

---

## Why I built it

When applying to many roles, it is hard to tell quickly which parts of a JD
your resume does not cover. Keyword checkers miss meaning ("built dashboards"
vs "data visualization"), and pure LLM tools give a different answer every
time. This app mixes both: a stable, explainable score plus optional LLM advice.

---

## How it works

```
Resume PDF ──► read text (pypdf) ──► clean ──► split into lines ─┐
                                                                 ├─► Meaning match (embeddings)  ┐
Job description ──────────────────► clean ──► split into lines ─┘                                ├─► Overall score
                                                                     Skill match (skill list)    ┘
                                                          (optional) Gemini ──► written feedback
```

### 1. Meaning match (0–100)

- The resume and the JD are split into short lines.
- Each line is turned into a vector with **`all-MiniLM-L6-v2`**, a small
  pretrained transformer from sentence-transformers.
- For **every JD line**, the app finds the **most similar resume line**
  (cosine similarity) and averages those best scores.
- Raw cosine scores for related text usually fall between about 0.2 and 0.8,
  so this range is stretched to 0–100.

Why line by line and not whole documents? The model only reads about 256
tokens at a time, so embedding a full resume would quietly ignore most of it.
Comparing line by line also tells us *which* JD lines are weakly covered.

### 2. Skill match (0–100)

- `skills.py` has a hand-made list of ~45 data/ML skills, each with common
  spellings (e.g. `scikit-learn`, `sklearn`).
- The app finds which skills appear in the JD and in the resume.
- Skill match = share of JD skills that also appear in the resume.
- Word boundaries are used, so "spark" inside "sparkling" does not count.

### 3. Overall score

```
overall = 0.5 × meaning match + 0.5 × skill match
```

### 4. LLM feedback (optional)

If a Gemini API key is set, the app sends the resume and JD to Gemini with a
short prompt that asks for three sections: missing skills, strong points and
quick fixes. The prompt tells the model to use only the given text and not
invent experience.

---

## Project structure

```
resume-jd-matcher/
├── app.py                 # Streamlit user interface
├── matcher.py             # PDF reading, cleaning, scoring
├── skills.py              # skill list + skill finder
├── llm.py                 # optional Gemini feedback
├── tests/test_matcher.py  # unit tests (no internet needed)
├── sample_data/sample_jd.txt
├── requirements.txt
└── .streamlit/secrets.toml.example
```

---

## Run it on your computer

```bash
git clone https://github.com/<your-username>/resume-jd-matcher.git
cd resume-jd-matcher
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the embedding model (about 90 MB). After that it is cached.

### Add the Gemini key (optional)

1. Get a free key at <https://aistudio.google.com/apikey>.
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
3. Paste your key there. This file is in `.gitignore`, so it will not be pushed to GitHub.

You can also set it as an environment variable: `export GEMINI_API_KEY=...`.
To use a different Gemini model, set `GEMINI_MODEL` (default: `gemini-3.8-flash`, with `gemini-flash-latest` as a backup).

### Run the tests

```bash
pytest -q
```

The tests use a tiny fake embedder, so they run in under a second with no download.

---

## Deploy on Streamlit Cloud (free)

1. Push this folder to a public GitHub repo.
2. Go to <https://share.streamlit.io>, click **New app** and pick the repo, with `app.py` as the main file.
3. Under **Advanced settings → Secrets**, add `GEMINI_API_KEY = "your-key"`.
4. Click **Deploy**, then paste the link at the top of this README.

---

## Example

Running it on my own resume against a Data Scientist JD showed a skill match of
about 42%. The skill list flagged **EDA, Feature Engineering, Data Cleaning and
Deployment** as missing, even though my projects do all of these. My resume
just never used those words. That is the kind of quick fix this tool is meant to catch.

---

## Limits (honest notes)

- The score is a **guide, not a verdict**. It does not know how good a project is.
- The skill list is fixed; skills not in `skills.py` are not counted. Add more as needed.
- Scanned (image) PDFs have no text, so they cannot be read. Use a text-based PDF.
- Very short lines (under 4 words) are dropped, so section titles do not affect the score.
- The 0.2–0.8 cosine range and the 50/50 weights are simple choices, not tuned on labelled data.

## Ideas for later

- Let the user change the 50/50 weight with a slider.
- Suggest which resume bullet to rewrite for each weak JD line.
- Compare one resume against several JDs at once.

## Tech used

Python · Streamlit · sentence-transformers (MiniLM) · NumPy · pypdf · Google Gemini API · pytest · Git
