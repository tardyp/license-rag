"""Pertinence scoring of a query text against one license text.

The unit of evidence is the shingle (see :mod:`license_rag.normalize`): every
score component answers a question a human reviewer would ask when deciding
whether a license text is *this* license:

- ``query_passage``: is the query a contiguous passage of the license? The
  longest run of the query's shingles that appears *in the same order and
  consecutively* in the license, as a fraction of the query. This is the
  decisive signal: it separates a text copied from this license from a text
  that merely reuses its words, since reordered or rewritten text breaks runs.
- ``query_coverage``: how much of the query appears in the license *anywhere*.
  Coverage survives rewording and reordering, which passage evidence does not,
  so it is what keeps a vendor-rewritten license variant at a usable
  pertinence instead of collapsing to "low".
- ``license_coverage``: how much of the license appears anywhere in the query.
  Distinguishes "here is the full GPL-3.0" from "here is a GPL-3.0 fragment".

All three are fractions in ``[0, 1]`` and independent of text length. Passage
evidence leads because a contiguous in-order copy is proof of derivation;
coverage follows as the tolerance for real-world edits and partial texts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from license_rag.normalize import SHINGLE_K, normalize_text, shingle_hashes, tokenize

# Weights of the three components.
QUERY_PASSAGE_WEIGHT = 0.40
QUERY_COVERAGE_WEIGHT = 0.35
LICENSE_COVERAGE_WEIGHT = 0.25

# Shingles need a query of at least this many tokens to exist at all; shorter
# "queries" (a bare license name, "MIT") carry no textual evidence and are
# resolved through the identifier channel instead.
MIN_QUERY_TOKENS = SHINGLE_K


@dataclass(frozen=True, slots=True)
class LexicalScore:
    """The lexical evidence for one (query, license) pair."""

    query_passage: float = 0.0
    query_coverage: float = 0.0
    license_coverage: float = 0.0
    matched_shingles: int = 0

    @property
    def similarity(self) -> float:
        """Return the lexical similarity in ``[0, 1]``."""
        return (
            QUERY_PASSAGE_WEIGHT * self.query_passage
            + QUERY_COVERAGE_WEIGHT * self.query_coverage
            + LICENSE_COVERAGE_WEIGHT * self.license_coverage
        )

    @property
    def is_covered(self) -> bool:
        """Return whether every shingle of the query is present in the license."""
        return self.query_coverage >= 1.0


def ordered_shingles(text: str, k: int = SHINGLE_K) -> np.ndarray:
    """Return the shingle sequence of ``text`` in text order, duplicates kept."""
    tokens = tokenize(normalize_text(text))
    if len(tokens) < k:
        return np.empty(0, dtype=np.uint64)
    return shingle_hashes(tokens, k=k, unique=False)


# Mixer used to combine two adjacent shingles into one hash. Applying it to the
# ordered shingle sequences of both the query and the license makes "these two
# shingles appear consecutively" a plain set membership test, which is what
# turns passage measurement into the same numpy operation as coverage.
BIGRAM_MIX = np.uint64(0xD6E8FEB86659FD93)


def bigram_hashes(sequence: np.ndarray) -> np.ndarray:
    """Return the hashes of consecutive shingle pairs of ``sequence``.

    A pair hash present in both texts means those six tokens appeared in that
    exact order in both, so a run of consecutive matched pairs in the query is a
    contiguous copy of the query inside the license -- and a reordering, however
    sentence-aligned, breaks the run at every block boundary.
    """
    if sequence.size < 2:
        return np.empty(0, dtype=np.uint64)
    return np.bitwise_xor(sequence[:-1] * BIGRAM_MIX, sequence[1:])


def _membership(sorted_values: np.ndarray, probe: np.ndarray) -> np.ndarray:
    """Return a bool mask of which ``probe`` values are present in ``sorted_values``."""
    if probe.size == 0 or sorted_values.size == 0:
        return np.zeros(probe.shape, dtype=bool)
    positions = np.searchsorted(sorted_values, probe)
    positions[positions == sorted_values.size] = sorted_values.size - 1
    return sorted_values[positions] == probe


def longest_run(mask: np.ndarray) -> tuple[int, int]:
    """Return ``(length, start)`` of the longest run of True values in ``mask``."""
    if not mask.any():
        return 0, -1
    padded = np.concatenate(([False], mask, [False]))
    transitions = np.diff(padded.view(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    lengths = ends - starts
    best = int(np.argmax(lengths))
    return int(lengths[best]), int(starts[best])


def score_pair(
    query_shingles: np.ndarray,
    query_bigrams: np.ndarray,
    target_bigrams: np.ndarray,
    target_shingles: np.ndarray,
) -> LexicalScore:
    """Return the lexical evidence of a query against one license.

    ``query_shingles``/``target_shingles`` are unique sorted shingle sets, used
    for coverage; ``query_bigrams``/``target_bigrams`` are the sets of
    consecutive shingle pairs, used for contiguity (see :func:`bigram_hashes`).
    """
    if query_shingles.size == 0 or target_shingles.size == 0:
        return LexicalScore()

    matched = int(np.count_nonzero(_membership(target_shingles, query_shingles)))
    if not matched:
        return LexicalScore()

    # Longest contiguous copy of the query inside the license, measured on
    # consecutive shingle pairs so that order is required, not just presence.
    passage = 0.0
    if query_bigrams.size and target_bigrams.size:
        run, _ = longest_run(_membership(target_bigrams, query_bigrams))
        passage = min(1.0, run / query_bigrams.size)

    return LexicalScore(
        query_passage=passage,
        query_coverage=matched / query_shingles.size,
        license_coverage=matched / target_shingles.size,
        matched_shingles=matched,
    )


def evidence_passage(query_shingles: np.ndarray, target_ordered: np.ndarray) -> tuple[int, int]:
    """Return ``(start, length)`` in tokens of the license text that matched.

    This is the longest contiguous passage of the license present in the query,
    reported so a reviewer can see the text the score is based on. It is
    computed only for the matches actually reported, never for every candidate.
    """
    if query_shingles.size == 0 or target_ordered.size == 0:
        return -1, 0
    return longest_run(_membership(query_shingles, target_ordered))[::-1]
