"""Text normalization and shingling.

License identification is a near-duplicate detection problem, so normalization
must be aggressive enough to make the same license text look identical across
whitespace, case, punctuation, quoting and unicode differences, but it must
never rewrite words: any token change is a real textual difference.

Two derived representations are produced for every text:

- ``tokenize``: the ordered word tokens.
- ``shingle_set``: the set of ``k`` consecutive tokens, hashed to uint64
  (``k=5`` by default). Shingles are the unit of matching: a query is scored by
  how much of it is covered by the shingles of a license text.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np

# Shingle width in tokens. 5 is the smallest width that is essentially unique
# for license boilerplate while still tolerating short quoted fragments; at
# k=5 a paraphrase that rewrites two adjacent words out of five still breaks
# the shingle, which is what makes the lexical channel precise.
SHINGLE_K = 5

_TOKEN_RE = re.compile(r"[0-9a-z]+")

# Characters that carry no meaning for license text and are stripped instead of
# becoming tokens. Unicode punctuation/dashes/quotes normalizes to ASCII first.
_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u200b": "",
        "\ufeff": "",
    }
)

# Odd hash values, one per shingle position, used to combine k word hashes into
# one shingle hash without materializing the k-gram strings.
_POSITION_MIX = tuple(
    np.uint64(x)
    for x in (
        0x9E3779B97F4A7C15,
        0xC2B2AE3D27D4EB4F,
        0x165667B19E3779F9,
        0x27D4EB2F165667C5,
        0x85EBCA77C2B2AE63,
        0xD6E8FEB86659FD93,
        0xA24BAED4963EE407,
        0x9FB21C651E98DF25,
    )
)


def normalize_text(text: str) -> str:
    """Return ``text`` lowercased, unicode-normalized and whitespace-collapsed.

    Idempotent. Used both for the stored searchable form of a license and for
    incoming queries and for the exact-equality shortcut.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).translate(_PUNCT_TRANSLATION)
    text = text.lower()
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    """Return the word tokens of ``text`` (already-normalized input is expected)."""
    return _TOKEN_RE.findall(text)


def word_hashes(tokens: list[str]) -> np.ndarray:
    """Return a uint64 hash per token, using a shared cache for repeated words."""
    cache: dict[str, int] = _WORD_HASH_CACHE
    out = np.empty(len(tokens), dtype=np.uint64)
    digest = hashlib.blake2b
    for i, token in enumerate(tokens):
        h = cache.get(token)
        if h is None:
            h = int.from_bytes(digest(token.encode("utf-8"), digest_size=8).digest(), "big")
            cache[token] = h
        out[i] = h
    return out


_WORD_HASH_CACHE: dict[str, int] = {}


def shingle_hashes(tokens: list[str], k: int = SHINGLE_K, unique: bool = True) -> np.ndarray:
    """Return the shingle hashes of ``tokens``, sorted when ``unique``.

    ``unique=True`` (default) returns the sorted unique set, the form used for
    set operations; ``unique=False`` keeps duplicates and text order, the form
    needed to measure whether a passage is contiguous. Empty (and texts shorter
    than ``k`` tokens) yield an empty array: those texts carry no reliable
    identity and must not match everything.
    """
    if len(tokens) < k:
        return np.empty(0, dtype=np.uint64)
    words = word_hashes(tokens)
    mixed = np.zeros(len(tokens) - k + 1, dtype=np.uint64)
    for position in range(k):
        mixed ^= words[position : len(tokens) - k + 1 + position] * _POSITION_MIX[position]
    if not unique:
        return mixed
    return np.unique(mixed)


def shingle_set(text: str, k: int = SHINGLE_K) -> np.ndarray:
    """Return the unique, sorted shingle hashes of a raw text."""
    return shingle_hashes(tokenize(normalize_text(text)), k=k)


def token_digest(text: str) -> str:
    """Return the sha1 of the token sequence of ``text``.

    Identical digests mean identical words in the same order, which is exactly
    the equivalence the scoring engine uses: texts that differ only in
    punctuation, quoting, whitespace or letter case are the same license text
    here, and must be grouped as one match rather than ranked against each other.
    """
    return hashlib.sha1(" ".join(tokenize(normalize_text(text))).encode("utf-8")).hexdigest()
