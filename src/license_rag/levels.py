"""Pertinence levels: turning retrieval scores into a verdict a human can act on.

A match carries two, separately interpretable components:

- a **lexical** similarity in ``[0, 1]``, the fraction of the query that is
  verbatim text of the license (see :mod:`license_rag.scoring`);
- a **semantic** similarity in ``[0, 1]``, cosine similarity of the static
  embeddings, rescaled with the constants measured below.

The lexical component dominates because it is evidence about the exact text
supplied. The semantic component only breaks ties between lexically similar
candidates and, when there is no verbatim evidence at all, produces a capped
"this reads like" answer that can never claim to be a confirmed match.

Constants were calibrated by measurement on the shipped corpus (see
``evals/evaluate.py``): unrelated license chunks score at most 0.864 cosine
(99.9th percentile of random pairs) while an excerpt against its own license
scores 0.969 at the median.
"""

from __future__ import annotations

# Level names in decreasing order of pertinence, with the minimum final score
# required to reach each level.
LEVEL_THRESHOLDS: tuple[tuple[str, float], ...] = (
    ("exact", 0.97),
    ("very-high", 0.88),
    ("high", 0.72),
    ("medium", 0.50),
    ("low", 0.30),
    ("none", 0.00),
)

LEVEL_ORDER: tuple[str, ...] = tuple(name for name, _ in LEVEL_THRESHOLDS)

# Semantic rescaling: cosine at or below FLOOR is noise, at or above CEILING is
# a confident semantic match.
SEMANTIC_FLOOR = 0.87
SEMANTIC_CEILING = 0.98

# A semantic-only match is a hint, not an identification: it is capped below
# the "high" level so that a reworded or summarized text is never reported as a
# confirmed license.
SEMANTIC_ONLY_CAP = 0.55

# Semantic agreement can lift a lexical score by at most this fraction of the
# remaining distance, keeping ranking stable when several licenses share text.
SEMANTIC_TIEBREAK = 0.30

# Score assigned to a match proven by identity rather than similarity: the query
# is the license text itself, or is the license's own identifier.
IDENTITY_SCORE = 1.0

# Score for a license named by an SPDX-License-Identifier tag inside the query.
# A tag is a declaration about the text rather than the text itself, so it ranks
# below an identified text but still counts as a strong, explicit match.
TAG_SCORE = 0.95


def fuse(lexical: float, semantic: float, headroom: float = 1.0) -> float:
    """Combine the lexical and semantic channels into one pertinence score.

    ``headroom`` in ``[0, 1]`` is how much of the query is *not* covered by a
    contiguous verbatim passage, and therefore how much the semantic channel can
    still contribute. When the query is fully present in the license (headroom
    0) the lexical evidence is saturated and semantic similarity is pure noise:
    every license derived from that license matches the passage equally well,
    and letting embedding similarity order them only shuffles an arbitrary
    choice. Semantic similarity earns its weight exactly on reworded text, where
    contiguity is gone but much of the wording remains.
    """
    lexical = min(max(lexical, 0.0), 1.0)
    semantic = min(max(semantic, 0.0), 1.0)
    headroom = min(max(headroom, 0.0), 1.0)
    if lexical <= 0.0:
        return SEMANTIC_ONLY_CAP * semantic
    return lexical + (1.0 - lexical) * SEMANTIC_TIEBREAK * semantic * headroom


def level_for(score: float) -> str:
    """Return the pertinence level name for a final ``[0, 1]`` score."""
    for name, threshold in LEVEL_THRESHOLDS:
        if score >= threshold:
            return name
    return LEVEL_THRESHOLDS[-1][0]


def pertinence(score: float) -> int:
    """Return the pertinence as an integer percentage in ``[0, 100]``."""
    return int(round(min(max(score, 0.0), 1.0) * 100))


def at_least(level: str, minimum: str) -> bool:
    """Return whether ``level`` is at least as pertinent as ``minimum``."""
    return LEVEL_ORDER.index(level) <= LEVEL_ORDER.index(minimum)
