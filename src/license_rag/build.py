"""Building the retrieval database from a ScanCode LicenseDB corpus.

Everything the search path needs is computed here once and stored:

- normalized text and shingle representations of every license text;
- the inverted index (shingle -> licenses) used to shortlist candidates
  without scanning the corpus;
- doc level and chunk level embeddings of every license.

The build is deterministic: the same corpus and model produce the same database
contents (only the recorded build timestamp changes).
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from license_rag.corpus import LicenseRecord, load_corpus
from license_rag.embed import CHUNK_STRIDE, CHUNK_TOKENS, DEFAULT_MODEL, Embedder, text_chunks
from license_rag.normalize import SHINGLE_K, normalize_text, token_digest
from license_rag.scoring import bigram_hashes, ordered_shingles
from license_rag.store import SCHEMA_VERSION, connect, create_schema

TOOL_VERSION = "0.1.0"

# Aliases are resolved in this order: a license that has an SPDX key wins the
# alias over one that does not, so "apache 2 0" resolves to the SPDX-annotated
# entry rather than to a LicenseDB-only variant.
_ALIAS_KIND_ORDER = ("key", "spdx", "spdx-other", "name", "short_name")


# -- building blocks ------------------------------------------------------


def corpus_digest(records: list[LicenseRecord]) -> str:
    """Return a digest of the corpus contents, to detect any corpus change."""
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda r: r.key):
        digest.update(record.key.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(token_digest(record.text).encode("ascii"))
        digest.update(b"\x00")
    return digest.hexdigest()


def build_postings(licenses: list[tuple[int, np.ndarray]]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build the inverted index from ``(license_id, unique shingles)`` pairs.

    Returns ``(shingles, posting_lists)`` where ``posting_lists[i]`` holds the
    ascending license ids of ``shingles[i]``.
    """
    shingle_parts = [shingles for _, shingles in licenses if shingles.size]
    id_parts = [
        np.full(shingles.size, license_id, dtype=np.int32) for license_id, shingles in licenses if shingles.size
    ]
    all_shingles = np.concatenate(shingle_parts)
    all_ids = np.concatenate(id_parts)

    order = np.lexsort((all_ids, all_shingles))
    all_shingles = all_shingles[order]
    all_ids = all_ids[order]

    boundaries = np.flatnonzero(all_shingles[1:] != all_shingles[:-1]) + 1
    keys = all_shingles[np.concatenate(([0], boundaries))]
    return keys, np.split(all_ids, boundaries)


def aliases_for(record: LicenseRecord) -> list[tuple[str, str]]:
    """Return the ``(alias, kind)`` pairs a license is addressable by."""
    pairs: list[tuple[str, str]] = []
    for key in (record.identifier, record.key):
        if key:
            pairs.append((key, "key"))
    if record.spdx_license_key:
        pairs.append((record.spdx_license_key, "spdx"))
    for other in record.other_spdx_license_keys:
        pairs.append((other, "spdx-other"))
    if record.name:
        pairs.append((record.name, "name"))
    if record.short_name:
        pairs.append((record.short_name, "short_name"))

    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for alias, kind in sorted(pairs, key=lambda pair: _ALIAS_KIND_ORDER.index(pair[1])):
        normalized = normalize_text(alias)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append((normalized, kind))
    return unique


# -- build ----------------------------------------------------------------


def build_database(
    corpus_path: str | Path,
    db_path: str | Path,
    model: str = DEFAULT_MODEL,
    with_embeddings: bool = True,
    corpus_source: str | None = None,
    log=print,
) -> dict:
    """Build the license-rag database and return its metadata.

    ``corpus_path`` is a LicenseDB ``docs`` directory (or a single license JSON
    file); ``db_path`` is overwritten. ``corpus_source`` records where the
    corpus came from (a downloaded revision, or a local path) in the database
    metadata.
    """
    started = time.time()
    db_path = Path(db_path)
    records, skipped = load_corpus(corpus_path)
    if not records:
        raise ValueError(f"no license with text found under {corpus_path}")
    log(f"corpus: {len(records)} licenses with text, {len(skipped)} skipped (no text)")

    # 1. text representations
    prepared = []
    for position, record in enumerate(records, start=1):
        norm_text = normalize_text(record.text)
        sequence = ordered_shingles(record.text)
        prepared.append((record, norm_text, sequence, np.unique(sequence), np.unique(bigram_hashes(sequence))))
        if position % 500 == 0:
            log(f"  shingled {position}/{len(records)} licenses")
    log(f"shingled corpus in {time.time() - started:.1f}s")

    # 2. embeddings
    doc_matrix = None
    chunk_vectors = None
    embedding_dim = 0
    if with_embeddings:
        embedder = Embedder(model)
        embed_started = time.time()
        doc_matrix = embedder.encode([norm_text for _, norm_text, _, _, _ in prepared])
        embedding_dim = int(doc_matrix.shape[1])
        log(f"embedded {doc_matrix.shape[0]} license texts in {time.time() - embed_started:.1f}s")

        chunk_texts: list[str] = []
        chunk_spans: list[tuple[int, int, int, int]] = []
        for license_id in range(1, len(prepared) + 1):
            for chunk_index, (chunk, start, end) in enumerate(text_chunks(prepared[license_id - 1][1])):
                chunk_texts.append(chunk)
                chunk_spans.append((license_id, chunk_index, start, end))
        chunk_started = time.time()
        chunk_vectors = embedder.encode(chunk_texts)
        log(
            f"embedded {len(chunk_texts)} chunks ({len(chunk_texts) / len(prepared):.1f}/license) "
            f"in {time.time() - chunk_started:.1f}s"
        )

    # 3. write
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = connect(db_path, readonly=False)
    try:
        create_schema(connection)
        with connection:
            connection.executemany(
                """INSERT INTO licenses (id, key, spdx_license_key, name, short_name, category, owner,
                        homepage_url, is_exception, is_deprecated, replaced_by, minimum_coverage,
                        n_tokens, n_shingles, token_digest)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        license_id,
                        record.key,
                        record.spdx_license_key,
                        record.name,
                        record.short_name,
                        record.category,
                        record.owner,
                        record.homepage_url,
                        int(record.is_exception),
                        int(record.is_deprecated),
                        json.dumps(list(record.replaced_by)),
                        record.minimum_coverage,
                        len(norm_text.split()),
                        int(shingles.size),
                        token_digest(record.text),
                    )
                    for license_id, (record, norm_text, _, shingles, _) in enumerate(prepared, start=1)
                ],
            )
            connection.executemany(
                "INSERT INTO texts (license_id, text, norm_text) VALUES (?,?,?)",
                [
                    (license_id, record.text, norm_text)
                    for license_id, (record, norm_text, _, _, _) in enumerate(prepared, start=1)
                ],
            )
            connection.executemany(
                "INSERT INTO shingle_data (license_id, shingle_set, shingle_seq, bigram_set) VALUES (?,?,?,?)",
                [
                    (license_id, shingles.tobytes(), sequence.tobytes(), bigrams.tobytes())
                    for license_id, (_, _, sequence, shingles, bigrams) in enumerate(prepared, start=1)
                ],
            )

            keys, posting_lists = build_postings(
                [(license_id, shingles) for license_id, (_, _, _, shingles, _) in enumerate(prepared, start=1)]
            )
            connection.executemany(
                "INSERT INTO postings (shingle, docs) VALUES (?,?)",
                [(key, docs.tobytes()) for key, docs in zip(keys.view(np.int64).tolist(), posting_lists, strict=True)],
            )
            log(f"indexed {keys.size} distinct shingles, {sum(len(d) for d in posting_lists)} postings")
            del keys, posting_lists

            if doc_matrix is not None:
                connection.executemany(
                    "INSERT INTO vectors (license_id, vec) VALUES (?,?)",
                    [
                        (license_id, doc_matrix[license_id - 1].astype(np.float32).tobytes())
                        for license_id in range(1, len(prepared) + 1)
                    ],
                )
                connection.executemany(
                    "INSERT INTO chunks (license_id, chunk_index, start_token, end_token, vec) VALUES (?,?,?,?,?)",
                    [
                        (license_id, chunk_index, start, end, chunk_vectors[row].astype(np.float32).tobytes())
                        for row, (license_id, chunk_index, start, end) in enumerate(chunk_spans)
                    ],
                )

            alias_rows: list[tuple[str, int, str]] = []
            for license_id, (record, _, _, _, _) in enumerate(prepared, start=1):
                for alias, kind in aliases_for(record):
                    alias_rows.append((alias, license_id, kind))
            connection.executemany(
                "INSERT INTO aliases (alias, license_id, kind) VALUES (?,?,?)",
                alias_rows,
            )

            connection.executemany(
                "INSERT INTO meta (key, value) VALUES (?,?)",
                [
                    ("schema_version", str(SCHEMA_VERSION)),
                    ("tool_version", TOOL_VERSION),
                    ("built_at", datetime.now(UTC).isoformat(timespec="seconds")),
                    ("corpus_path", str(Path(corpus_path).resolve())),
                    ("corpus_source", corpus_source or str(Path(corpus_path).resolve())),
                    ("corpus_digest", corpus_digest(records)),
                    ("license_count", str(len(prepared))),
                    ("skipped_count", str(len(skipped))),
                    ("skipped_keys", json.dumps(skipped)),
                    ("shingle_k", str(SHINGLE_K)),
                    ("embedding_model", model if with_embeddings else ""),
                    ("embedding_dim", str(embedding_dim)),
                    ("chunk_tokens", str(CHUNK_TOKENS)),
                    ("chunk_stride", str(CHUNK_STRIDE)),
                    ("build_seconds", f"{time.time() - started:.1f}"),
                ],
            )
        connection.execute("ANALYZE")
        connection.execute("VACUUM")
    finally:
        connection.close()

    log(f"built {db_path} ({db_path.stat().st_size / 1e6:.1f} MB) in {time.time() - started:.1f}s")
    return {
        "db_path": str(db_path),
        "licenses": len(prepared),
        "skipped": len(skipped),
        "embedding_model": model if with_embeddings else "",
        "build_seconds": round(time.time() - started, 1),
    }
