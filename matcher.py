"""Core logic: read the resume, compare it with the JD, and build a score.

The final score mixes two simple parts:

1. Meaning match (semantic score): how close the resume and the JD are
   in meaning, using sentence embeddings.
2. Skill match: what share of the skills asked for in the JD also
   appear in the resume.

    final score = 0.5 * meaning match + 0.5 * skill match
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from skills import find_skills

# An "embedder" is any function that turns a list of texts into a 2D array
# (one row per text). In the app this is a sentence-transformers model.
# In tests we pass a tiny fake one, so tests run without downloads.
Embedder = Callable[[Sequence[str]], np.ndarray]

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Raw cosine scores from MiniLM for related documents usually fall
# between about 0.2 and 0.8, so we stretch that range to 0-100.
COS_LOW, COS_HIGH = 0.2, 0.8


@dataclass
class MatchResult:
    final_score: float
    semantic_score: float
    skill_score: float
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    extra_skills: list[str] = field(default_factory=list)
    weak_jd_lines: list[tuple[str, float]] = field(default_factory=list)


# ---------- reading and cleaning ----------

def read_pdf(file) -> str:
    """Read all text from a PDF (a path or a file-like object)."""
    from pypdf import PdfReader

    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def clean_text(text: str) -> str:
    """Fix common PDF issues: odd bullets, extra spaces, broken lines."""
    text = text.replace("•", "\n").replace("●", "\n").replace("", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def split_into_chunks(text: str, min_words: int = 4, max_words: int = 60) -> list[str]:
    """Split text into short pieces (lines / sentences) for embedding.

    Very short lines (like "GitHub" or a date) are dropped. Long lines are
    cut so they fit the model's input limit.
    """
    pieces = re.split(r"\n|(?<=[.!?])\s+", text)
    chunks = []
    for piece in pieces:
        words = piece.split()
        if len(words) < min_words:
            continue
        for i in range(0, len(words), max_words):
            chunks.append(" ".join(words[i : i + max_words]))
    return chunks


# ---------- scoring ----------

def load_embedder(model_name: str = MODEL_NAME) -> Embedder:
    """Load the sentence-transformers model and return an embed function."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts: Sequence[str]) -> np.ndarray:
        return model.encode(list(texts), normalize_embeddings=True)

    return embed


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def semantic_score(resume_chunks: list[str], jd_chunks: list[str], embed: Embedder):
    """For every JD line, find the closest resume line; average those.

    This is better than embedding each whole document once, because the
    model only reads about 256 tokens and would ignore most of a resume.

    Returns (score from 0 to 100, list of (jd_line, best_similarity)).
    """
    if not resume_chunks or not jd_chunks:
        return 0.0, []

    r = _normalize(np.asarray(embed(resume_chunks), dtype=float))
    j = _normalize(np.asarray(embed(jd_chunks), dtype=float))

    sims = j @ r.T                # rows = JD lines, cols = resume lines
    best = sims.max(axis=1)       # best resume match for each JD line

    scaled = np.clip((best - COS_LOW) / (COS_HIGH - COS_LOW), 0, 1)
    per_line = list(zip(jd_chunks, best.round(3).tolist()))
    return float(scaled.mean() * 100), per_line


def skill_score(resume_text: str, jd_text: str):
    """Compare skills found in the JD with skills found in the resume."""
    jd_skills = find_skills(jd_text)
    resume_skills = find_skills(resume_text)

    matched = sorted(jd_skills & resume_skills)
    missing = sorted(jd_skills - resume_skills)
    extra = sorted(resume_skills - jd_skills)

    score = 100.0 * len(matched) / len(jd_skills) if jd_skills else 0.0
    return score, matched, missing, extra


def match(resume_text: str, jd_text: str, embed: Embedder, w_semantic: float = 0.5) -> MatchResult:
    """Run the full comparison and return a MatchResult."""
    resume_text = clean_text(resume_text)
    jd_text = clean_text(jd_text)

    sem, per_line = semantic_score(split_into_chunks(resume_text), split_into_chunks(jd_text), embed)
    sk, matched, missing, extra = skill_score(resume_text, jd_text)

    final = w_semantic * sem + (1 - w_semantic) * sk
    weak = sorted(per_line, key=lambda x: x[1])[:5]  # JD lines the resume covers least

    return MatchResult(
        final_score=round(final, 1),
        semantic_score=round(sem, 1),
        skill_score=round(sk, 1),
        matched_skills=matched,
        missing_skills=missing,
        extra_skills=extra,
        weak_jd_lines=weak,
    )
