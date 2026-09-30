"""The semantic channel: static embeddings of license texts and text chunks.

License identification is anchored on verbatim evidence (see
:mod:`license_rag.scoring`); embeddings exist to cover the two cases where
verbatim evidence is weak or absent:

- only a short passage of a long license is supplied, where the passage is the
  query and the license is far longer (chunk level retrieval);
- the license has been reworded, translated or summarized, so no shingle
  survives (both levels).

Embeddings are therefore stored at two granularities per license: the whole
normalized text (truncated by the model) and overlapping token chunks. A query
is scored by the best of the two, which is what makes both long and short
queries work.
"""

from __future__ import annotations

import numpy as np

from license_rag.normalize import normalize_text, tokenize

# Default static embedding model: 256 dimensions, ~32MB, no GPU, no torch.
# Chosen by measurement over this corpus against potion-base-32M: the larger
# model improved chunk level top-1 by 2.5 to 5 points for 4x the size and cost.
DEFAULT_MODEL = "minishlab/potion-base-8M"

# Chunking of license texts for embedding: 160 tokens (~1.2KB) with 50%
# overlap. Chunks must be large enough to carry a full clause and small enough
# that a single vector is not an average of the whole license.
CHUNK_TOKENS = 160
CHUNK_STRIDE = 80
MIN_CHUNK_TOKENS = 25

# Chunks referencing a larger vocabulary are embedded in batches of this size.
BATCH_SIZE = 512


class Embedder:
    """Encodes text to L2-normalized float32 vectors (cosine = dot product)."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Load the static model on first use: a lexically built database needs no model."""
        if self._model is None:
            try:
                from model2vec import StaticModel
            except ImportError as error:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "the semantic channel needs the 'model2vec' package; install it or build/query with --no-embed"
                ) from error
            self._model = StaticModel.from_pretrained(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.dim)

    def encode(self, texts: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        """Return the normalized embedding of each text as a ``(len(texts), dim)`` array."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(texts, batch_size=batch_size, use_multiprocessing=False)
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        return vectors / norms

    def encode_one(self, text: str) -> np.ndarray:
        """Return the normalized embedding of a single text as a 1D array."""
        return self.encode([text])[0]


def text_chunks(text: str, size: int = CHUNK_TOKENS, stride: int = CHUNK_STRIDE) -> list[tuple[str, int, int]]:
    """Split ``text`` into overlapping token chunks.

    Returns ``(chunk_text, start_token, end_token)`` triples over the normalized
    token sequence, so a chunk can be traced back to the passage it came from.
    """
    tokens = tokenize(normalize_text(text))
    if not tokens:
        return []
    if len(tokens) <= size:
        return [(" ".join(tokens), 0, len(tokens))]

    chunks: list[tuple[str, int, int]] = []
    for start in range(0, len(tokens), stride):
        piece = tokens[start : start + size]
        if len(piece) < MIN_CHUNK_TOKENS:
            break
        chunks.append((" ".join(piece), start, start + len(piece)))
        if start + size >= len(tokens):
            break
    return chunks
