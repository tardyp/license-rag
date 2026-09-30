"""Query pipeline: arbitrary license text in, ranked SPDX identifiers out.

The pipeline is deliberately staged so that the expensive, corpus-wide signals
are only computed where they can matter:

1. **identity** - the query is a known identifier or name (``Apache-2.0``), or
   its normalized text is a license text already in the database. Answered
   directly from the alias and digest indexes.
2. **candidates** - the inverted index is probed with the query's shingles to
   shortlist licenses sharing verbatim text, and the embedding matrices are
   probed to shortlist licenses that are semantically close. Neither probe
   scans the corpus.
3. **verification** - every candidate is scored against the query by
   :func:`license_rag.scoring.score_pair`, which measures coverage in both
   directions and locates the matching passage.
4. **pertinence** - lexical and semantic evidence are fused
   (:mod:`license_rag.levels`) into a score and a level.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from license_rag import levels
from license_rag.embed import Embedder
from license_rag.normalize import SHINGLE_K, normalize_text, shingle_hashes, token_digest, tokenize
from license_rag.scoring import (
    MIN_QUERY_TOKENS,
    LexicalScore,
    bigram_hashes,
    evidence_passage,
    ordered_shingles,
    score_pair,
)
from license_rag.store import LicenseStore

# Licenses sharing fewer shingles than this with the query are not verified:
# one shared shingle is not evidence of anything.
MIN_SHARED_SHINGLES = 2

# Upper bounds on the work of a single query: candidates shortlisted by the
# inverted index and leaders of the semantic ranking. Both are bounded by the
# cost of verification (a mask over each candidate's shingle sequence), not by
# corpus size, so they stay fixed as the corpus grows.
MAX_LEXICAL_CANDIDATES = 150
MAX_SEMANTIC_CANDIDATES = 60

# Two licenses whose shingle sets overlap this much are the same license text
# stored under two keys (lgpl-2.1 and lgpl-2.1-plus differ only by the "or
# later" clause; ibm-icu and unicode-icu-58 by a version string). Such pairs are
# reported as one match, with the twins listed in ``also_matches``, because
# choosing between them is a corpus metadata question, not an identification
# result. Distinct licenses that merely share boilerplate stay well below this:
# mit against mit-0 measures 0.82, artistic-perl-1.0 against openldap-1.1 0.79.
TWIN_CONTAINMENT = 0.95
TWIN_SCORE_DELTA = 0.05
TWIN_LIMIT = 25

# Ranking band: matches whose scores are within this distance of the band's best
# score are treated as equally supported by evidence and ordered by preference
# instead. A fragment of a license is verbatim in every license derived from it
# (the Apache-2.0 definitions passage is verbatim in ~200 Apache-derived
# licenses), so at that distance the score cannot distinguish them; preferring
# the canonical, non-deprecated, non-exception, SPDX-identified license is what
# makes the reported identifier useful rather than arbitrary.
RANKING_BAND = 0.02

# Characters of the matched passage returned as evidence.
PASSAGE_CHARS = 320


@dataclass(frozen=True, slots=True)
class Match:
    """One candidate license for a query, with its evidence and pertinence."""

    identifier: str
    key: str
    score: float
    level: str
    pertinence: int
    name: str = ""
    short_name: str = ""
    category: str = ""
    owner: str = ""
    spdx_license_key: str | None = None
    is_exception: bool = False
    is_deprecated: bool = False
    replaced_by: tuple[str, ...] = ()
    lexical: LexicalScore = field(default_factory=LexicalScore)
    semantic: float = 0.0
    matched_passage: str = ""
    also_matches: tuple[str, ...] = ()
    reason: str = ""

    def as_dict(self) -> dict:
        """Return a JSON-serializable view of the match."""
        return {
            "identifier": self.identifier,
            "key": self.key,
            "score": round(self.score, 4),
            "pertinence": self.pertinence,
            "level": self.level,
            "name": self.name,
            "short_name": self.short_name,
            "category": self.category,
            "owner": self.owner,
            "spdx_license_key": self.spdx_license_key,
            "is_exception": self.is_exception,
            "is_deprecated": self.is_deprecated,
            "replaced_by": list(self.replaced_by),
            "evidence": {
                "reason": self.reason,
                "matched_shingles": self.lexical.matched_shingles,
                "query_passage": round(self.lexical.query_passage, 4),
                "query_coverage": round(self.lexical.query_coverage, 4),
                "license_coverage": round(self.lexical.license_coverage, 4),
                "lexical_score": round(self.lexical.similarity, 4),
                "semantic_score": round(self.semantic, 4),
                "matched_passage": self.matched_passage,
            },
            "also_matches": list(self.also_matches),
        }


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The outcome of one query."""

    query_tokens: int
    query_shingles: int
    considered: int
    matches: tuple[Match, ...]
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "query_tokens": self.query_tokens,
            "query_shingles": self.query_shingles,
            "considered": self.considered,
            "note": self.note,
            "matches": [match.as_dict() for match in self.matches],
        }


SPDX_TAG_RE = re.compile(
    r"spdx[\s\-_]*license[\s\-_]*identifier\s*[:=]\s*(?P<expression>[^\n\r*#]+)",
    re.IGNORECASE,
)

# Operators of an SPDX license expression, which separate identifiers that each
# resolve on their own.
_EXPRESSION_SPLIT_RE = re.compile(r"\s+(?:AND|OR|WITH)\s+|[()]+", re.IGNORECASE)


def spdx_tag_identifiers(text: str) -> list[str]:
    """Return the license identifiers declared by SPDX tags inside ``text``.

    Licenses reach a scanner as source files whose header declares
    ``SPDX-License-Identifier: MIT`` (or a full expression) far more often than
    as a bare identifier, so the tag is resolved as its own evidence channel.
    """
    identifiers: list[str] = []
    for match in SPDX_TAG_RE.finditer(text):
        for part in _EXPRESSION_SPLIT_RE.split(match.group("expression")):
            part = part.strip()
            if part:
                identifiers.append(part)
    return identifiers


class Searcher:
    """Queries a license-rag database.

    Reuse one instance across queries: it caches the license metadata, the
    embedding matrices and the alias index, which is where almost all of the
    per-query cost would otherwise go.
    """

    def __init__(self, db_path: str | Path, embedder: Embedder | None = None, use_embeddings: bool = True):
        self.store = LicenseStore(db_path)
        self.use_embeddings = use_embeddings and self.store.has_vectors
        self.embedder = (
            (embedder or Embedder(self.store.meta.get("embedding_model") or None)) if self.use_embeddings else None
        )
        self._digest_index: dict[str, list[int]] | None = None

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Searcher:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- channels ---------------------------------------------------------

    def _identity_matches(self, query_norm: str, raw_text: str) -> dict[int, tuple[str, float]]:
        """Return ``license_id -> (reason, score)`` for direct identity evidence.

        Three sources: the query is an identifier or name of a license, the
        query's normalized text is a license text already in the database, or the
        query carries an SPDX-License-Identifier tag naming a license.
        """
        hits: dict[int, tuple[str, float]] = {}
        for license_id, kind in self.store.aliases.get(query_norm, ()):
            hits[license_id] = (f"identifier match ({kind})", levels.IDENTITY_SCORE)
        if self._digest_index is None:
            self._digest_index = self.store.digest_index()
        for license_id in self._digest_index.get(token_digest(query_norm), ()):
            hits.setdefault(license_id, ("identical text", levels.IDENTITY_SCORE))
        for identifier in spdx_tag_identifiers(raw_text):
            for license_id, kind in self.store.aliases.get(normalize_text(identifier), ()):
                hits.setdefault(
                    license_id,
                    (f"SPDX-License-Identifier tag ({identifier}, matched the {kind})", levels.TAG_SCORE),
                )
        return hits

    def _lexical_candidates(self, query_shingles: np.ndarray) -> dict[int, int]:
        """Return ``license_id -> shared shingle count`` from the inverted index."""
        parts = [license_ids for _, license_ids in self.store.postings(query_shingles) if license_ids.size]
        if not parts:
            return {}
        all_ids = np.concatenate(parts)
        if all_ids.size == 0:
            return {}
        license_ids, counts = np.unique(all_ids, return_counts=True)
        minimum = MIN_SHARED_SHINGLES if query_shingles.size >= MIN_QUERY_TOKENS else 1
        keep = counts >= minimum
        return dict(zip(license_ids[keep].tolist(), counts[keep].tolist(), strict=True))

    def _shortlist(self, counts: dict[int, int], semantic: np.ndarray) -> list[int]:
        """Return the license ids to verify: the strongest lexical hits plus the semantic leaders.

        The lexical shortlist preserves ties at the cut-off, because licenses of
        the same family routinely share the exact same number of shingles with a
        short query (a 25 shingle MIT header matches every MIT variant equally),
        and the tie is only broken by the verification score. Ties are ordered
        by semantic similarity so that the cut-off, when it must cut, prefers
        the semantically closest of the equally-covered candidates.
        """
        if len(counts) > MAX_LEXICAL_CANDIDATES:
            threshold = sorted(counts.values(), reverse=True)[MAX_LEXICAL_CANDIDATES - 1]
            lexical = [license_id for license_id, count in counts.items() if count >= threshold]
        else:
            lexical = list(counts)
        lexical.sort(key=lambda license_id: (-counts[license_id], -float(semantic[license_id]), license_id))
        lexical = lexical[:MAX_LEXICAL_CANDIDATES]

        semantic_order = np.argsort(-semantic)[:MAX_SEMANTIC_CANDIDATES]
        shortlist = list(
            dict.fromkeys(lexical + [int(license_id) for license_id in semantic_order if semantic[license_id] > 0])
        )
        return shortlist

    def _semantic_scores(self, query_norm: str) -> np.ndarray:
        """Return the semantic pertinence of every license, indexed by license id.

        The raw cosine of the best matching vector (whole license or chunk) is
        rescaled by :mod:`license_rag.levels`' measured floor and ceiling: below
        the floor it is noise between unrelated licenses, above the ceiling it is
        a confident semantic match. The rescale is vectorized here rather than
        called per license because it runs over the whole corpus at once.
        """
        scores = np.zeros(self.store.count + 1, dtype=np.float32)
        if not self.use_embeddings:
            return scores
        query_vector = self.embedder.encode_one(query_norm)
        best = np.full(self.store.count + 1, -1.0, dtype=np.float32)

        # Doc vectors are one row per license in id order: direct assignment.
        doc_ids, doc_matrix = self.store.doc_vectors()
        best[doc_ids] = doc_matrix @ query_vector

        # Chunks are grouped per license in id order, so the per-license maximum
        # is a segmented reduction over the similarity vector.
        _, chunk_matrix = self.store.chunk_vectors()
        segment_ids, segment_starts = self.store.chunk_segments()
        chunk_max = np.maximum.reduceat(chunk_matrix @ query_vector, segment_starts)
        np.maximum.at(best, segment_ids, chunk_max)

        floor, ceiling = levels.SEMANTIC_FLOOR, levels.SEMANTIC_CEILING
        return np.clip((best - floor) / (ceiling - floor), 0.0, 1.0).astype(np.float32)

    def _reason(self, lexical: LexicalScore, semantic: float, exact: bool) -> str:
        if exact:
            return "the supplied text is this license"
        if lexical.matched_shingles == 0 and semantic <= 0:
            return "no evidence"
        parts = []
        if lexical.query_passage > 0:
            parts.append(f"{self._percent(lexical.query_passage)} of the query is a contiguous passage of this license")
        if lexical.query_coverage > lexical.query_passage:
            parts.append(f"{self._percent(lexical.query_coverage)} of the query matches text of this license")
        if lexical.license_coverage > 0:
            parts.append(f"{self._percent(lexical.license_coverage)} of this license is present in the query")
        if semantic > 0:
            parts.append(f"semantic similarity {self._percent(semantic)}")
        return "; ".join(parts)

    @staticmethod
    def _percent(value: float) -> str:
        return f"{round(min(max(value, 0.0), 1.0) * 100)}%"

    # -- query ------------------------------------------------------------

    def search(
        self,
        text: str,
        top_k: int = 5,
        min_level: str = "none",
        include_duplicates: bool = False,
    ) -> SearchResult:
        """Return the best matching licenses for ``text``, best first.

        Matches whose license texts are identical, and matches whose texts are
        near-identical (the same license stored under two keys), are reported as
        one match with the other identifiers listed in ``also_matches`` unless
        ``include_duplicates`` is set.
        """
        query_norm = normalize_text(text)
        tokens = tokenize(query_norm)
        if not tokens:
            return SearchResult(0, 0, 0, (), "empty query")

        identity = self._identity_matches(query_norm, text)
        query_shingles = shingle_hashes(tokens)
        query_bigrams = bigram_hashes(ordered_shingles(query_norm))
        semantic = (
            self._semantic_scores(query_norm)
            if self.use_embeddings
            else np.zeros(self.store.count + 1, dtype=np.float32)
        )

        candidates = self._lexical_candidates(query_shingles) if query_shingles.size else {}
        candidate_ids = self._shortlist(candidates, semantic)
        candidate_ids = list(dict.fromkeys(candidate_ids + list(identity)))

        scored: list[Match] = []
        shingle_pairs = self.store.shingle_pairs(candidate_ids)
        for license_id in candidate_ids:
            row = self.store.license(license_id)
            target_shingles, target_ordered, target_bigrams = shingle_pairs[license_id]
            if query_shingles.size:
                lexical = score_pair(
                    query_shingles,
                    query_bigrams,
                    target_bigrams,
                    target_shingles,
                )
            else:
                lexical = LexicalScore()
            semantic_score = float(semantic[license_id])
            reason, identity_score = identity.get(license_id, ("", 0.0))
            identity_hit = bool(reason)
            score = (
                identity_score
                if identity_hit
                else levels.fuse(
                    lexical.similarity,
                    semantic_score,
                    headroom=1.0 - lexical.query_passage,
                )
            )
            if not reason:
                reason = self._reason(lexical, semantic_score, identity_hit)

            scored.append(
                Match(
                    identifier=row["spdx_license_key"] or row["key"],
                    key=row["key"],
                    score=score,
                    level=levels.level_for(score),
                    pertinence=levels.pertinence(score),
                    name=row["name"],
                    short_name=row["short_name"],
                    category=row["category"],
                    owner=row["owner"],
                    spdx_license_key=row["spdx_license_key"],
                    is_exception=bool(row["is_exception"]),
                    is_deprecated=bool(row["is_deprecated"]),
                    replaced_by=tuple(json_loads(row["replaced_by"])),
                    lexical=lexical,
                    semantic=semantic_score,
                    reason=reason,
                )
            )

        scored.sort(key=lambda match: (-match.score, -match.lexical.license_coverage, match.identifier))
        scored = self._rank(scored)
        if not include_duplicates:
            scored = self._merge_duplicates(scored)
            scored = self._merge_twins(scored, shingle_pairs)

        selected = [match for match in scored if levels.at_least(match.level, min_level)][:top_k]
        matches = tuple(self._with_passage(match, query_shingles, shingle_pairs) for match in selected)
        note = ""
        if query_shingles.size == 0:
            note = f"query is shorter than {MIN_QUERY_TOKENS} tokens: matched by identifier/semantics only"
        return SearchResult(len(tokens), int(query_shingles.size), len(candidate_ids), matches, note)

    def _with_passage(
        self,
        match: Match,
        query_shingles: np.ndarray,
        shingle_pairs: dict[int, tuple[np.ndarray, np.ndarray]],
    ) -> Match:
        """Return ``match`` with the passage of the license that matched filled in.

        Reconstructing the passage costs a mask over the license shingle
        sequence, so it is done only for the matches actually reported rather
        than for every candidate that was scored.
        """
        if not match.matched_passage:
            row = self.store.licenses_by_key.get(match.key)
            if row is not None and query_shingles.size:
                target_ordered = shingle_pairs[row["id"]][1]
                start, length = evidence_passage(query_shingles, target_ordered)
                match = replace(match, matched_passage=self._passage_text(row["id"], start, length))
        return match

    def _passage_text(self, license_id: int, start: int, length: int) -> str:
        """Return the license text at shingle ``start`` covering ``length`` shingles."""
        if start < 0 or length <= 0:
            return ""
        _, norm_text = self.store.text(license_id)
        tokens = tokenize(norm_text)
        end = min(len(tokens), start + length + SHINGLE_K - 1)
        return " ".join(tokens[start:end])[:PASSAGE_CHARS]

    @staticmethod
    def _rank(scored: list[Match]) -> list[Match]:
        """Order matches by score, and by preference within ``RANKING_BAND`` of the band's best.

        Matches at full confidence (the query is a license text, an identifier or
        an exact copy) are never reordered: only scores that fall short of proof
        are close enough for the score difference to be unsupported by evidence.
        Bands are grown greedily from the best remaining score, so the ordering
        never inverts two matches further apart than the band.
        """
        proven = sorted(
            (match for match in scored if match.score >= levels.IDENTITY_SCORE),
            key=lambda match: (-match.score, match.identifier),
        )
        ranked: list[Match] = []
        band: list[Match] = []
        for match in sorted(
            (match for match in scored if match.score < levels.IDENTITY_SCORE),
            key=lambda item: (-item.score, item.identifier),
        ):
            if band and band[0].score - match.score > RANKING_BAND:
                ranked.extend(sorted(band, key=Searcher._preference))
                band = []
            band.append(match)
        ranked.extend(sorted(band, key=Searcher._preference))
        return proven + ranked

    @staticmethod
    def _preference(match: Match) -> tuple:
        """Return the ordering key used inside a ranking band.

        Only licenses SPDX actually lists are preferred: ScanCode stores
        non-SPDX licenses under ``LicenseRef-scancode-*`` identifiers, and
        licenses with no SPDX key at all are reported under their LicenseDB key,
        which often looks like a plain identifier.
        """
        spdx_key = match.spdx_license_key or ""
        return (
            int(match.is_deprecated),
            int(match.is_exception),
            0 if spdx_key and not spdx_key.startswith("LicenseRef") else 1,
            -match.lexical.license_coverage,
            match.identifier,
        )

    def _merge_duplicates(self, matches: list[Match]) -> list[Match]:
        """Merge matches whose license texts are identical, keeping the best placed one."""
        digests = {row["key"]: row["token_digest"] for row in self.store.licenses.values()}
        merged: list[Match] = []
        seen: dict[str, int] = {}
        for match in matches:
            digest = digests.get(match.key, match.key)
            position = seen.get(digest)
            if position is None:
                seen[digest] = len(merged)
                merged.append(match)
                continue
            winner = merged[position]
            merged[position] = replace(winner, also_matches=winner.also_matches + (match.identifier,))
        return merged

    def _merge_twins(
        self,
        matches: list[Match],
        shingle_pairs: dict[int, tuple[np.ndarray, np.ndarray]],
    ) -> list[Match]:
        """Merge matches whose license texts are near-identical (see ``TWIN_CONTAINMENT``)."""
        heads, tail = matches[:TWIN_LIMIT], matches[TWIN_LIMIT:]
        sets: dict[str, np.ndarray] = {}

        def shingle_set(match: Match) -> np.ndarray:
            if match.key not in sets:
                row = self.store.licenses_by_key.get(match.key)
                pair = shingle_pairs.get(row["id"]) if row is not None else None
                sets[match.key] = pair[0] if pair else np.empty(0, dtype=np.uint64)
            return sets[match.key]

        merged: list[Match] = []
        for match in heads:
            target = shingle_set(match)
            for position, kept in enumerate(merged):
                if abs(kept.score - match.score) > TWIN_SCORE_DELTA:
                    continue
                reference = shingle_set(kept)
                if not reference.size or not target.size:
                    continue
                shared = np.intersect1d(reference, target, assume_unique=True).size
                if shared / min(reference.size, target.size) >= TWIN_CONTAINMENT:
                    merged[position] = replace(
                        kept,
                        also_matches=tuple(dict.fromkeys(kept.also_matches + (match.identifier,))),
                    )
                    break
            else:
                merged.append(match)
        return merged + tail


def json_loads(value, default=()):
    """Return the JSON-decoded ``value``, or ``default`` when empty or invalid."""
    if not value:
        return default
    try:
        return tuple(json.loads(value))
    except (ValueError, TypeError):
        return default
