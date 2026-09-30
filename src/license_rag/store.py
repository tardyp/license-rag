"""The retrieval database: a self-contained SQLite file.

Layout (all arrays are raw little-endian numpy buffers, so reads are zero-copy):

============================  =================================================
``meta``                      build parameters: schema version, shingle width,
                              window size, embedding model and dimensions,
                              corpus path, license count, build time
``licenses``                  one row per license: identifiers, metadata, sizes
``texts``                     the license text and its normalized form
``shingle_sets``              unique sorted uint64 shingles (set operations)
``shingle_seqs``              text-ordered uint64 shingles (window coverage)
``postings``                  inverted index shingle -> ascending int32 license ids
``vectors``                   doc level embedding per license
``chunks``                    chunk level embeddings, traceable to token spans
``aliases``                   identifier/name -> license, for direct lookups
============================  =================================================

The database is built by :mod:`license_rag.build`; it is queried through
:class:`license_rag.search.Searcher`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE licenses (
    id                  INTEGER PRIMARY KEY,
    key                 TEXT NOT NULL UNIQUE,
    spdx_license_key    TEXT,
    name                TEXT NOT NULL DEFAULT '',
    short_name          TEXT NOT NULL DEFAULT '',
    category            TEXT NOT NULL DEFAULT '',
    owner               TEXT NOT NULL DEFAULT '',
    homepage_url        TEXT NOT NULL DEFAULT '',
    is_exception        INTEGER NOT NULL DEFAULT 0,
    is_deprecated       INTEGER NOT NULL DEFAULT 0,
    replaced_by         TEXT NOT NULL DEFAULT '',
    minimum_coverage    INTEGER,
    n_tokens            INTEGER NOT NULL,
    n_shingles          INTEGER NOT NULL,
    token_digest        TEXT NOT NULL
);
CREATE INDEX licenses_digest ON licenses(token_digest);

CREATE TABLE texts (
    license_id  INTEGER PRIMARY KEY REFERENCES licenses(id),
    text        TEXT NOT NULL,
    norm_text   TEXT NOT NULL
);

CREATE TABLE shingle_data (
    license_id   INTEGER PRIMARY KEY REFERENCES licenses(id),
    shingle_set  BLOB NOT NULL,
    shingle_seq  BLOB NOT NULL,
    bigram_set   BLOB NOT NULL
);

CREATE TABLE postings (
    shingle  INTEGER PRIMARY KEY,
    docs     BLOB NOT NULL
) WITHOUT ROWID;

CREATE TABLE vectors (
    license_id  INTEGER PRIMARY KEY REFERENCES licenses(id),
    vec         BLOB NOT NULL
);

CREATE TABLE chunks (
    id           INTEGER PRIMARY KEY,
    license_id   INTEGER NOT NULL REFERENCES licenses(id),
    chunk_index  INTEGER NOT NULL,
    start_token  INTEGER NOT NULL,
    end_token    INTEGER NOT NULL,
    vec          BLOB NOT NULL
);
CREATE INDEX chunks_license ON chunks(license_id);

CREATE TABLE aliases (
    alias       TEXT NOT NULL,
    license_id  INTEGER NOT NULL REFERENCES licenses(id),
    kind        TEXT NOT NULL,
    PRIMARY KEY (alias, license_id)
) WITHOUT ROWID;
CREATE INDEX aliases_license ON aliases(license_id);
"""


def connect(path: str | Path, readonly: bool = True, timeout: float = 30.0) -> sqlite3.Connection:
    """Open the database, read-only by default."""
    path = Path(path)
    if readonly:
        if not path.exists():
            raise FileNotFoundError(f"no license-rag database at {path}: build one first")
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout)
    else:
        connection = sqlite3.connect(path, timeout=timeout)
    connection.row_factory = sqlite3.Row
    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    """Create an empty schema. Raises if tables already exist."""
    connection.executescript(DDL)


def _as_int64(array: np.ndarray) -> np.ndarray:
    """Reinterpret a uint64 shingle array as the signed int64 SQLite stores."""
    return array.view(np.int64)


def _as_uint64(value: int) -> np.uint64:
    """Reinterpret a signed int64 SQLite key as the uint64 shingle hash."""
    return np.uint64(value & 0xFFFFFFFFFFFFFFFF)


class LicenseStore:
    """Read access to a built license-rag database.

    Small tables (metadata, vectors, aliases) are loaded once and cached; the
    large per-license shingle blobs are fetched on demand for the candidates of
    a query only, which keeps memory proportional to the query, not the corpus.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.connection = connect(self.path, readonly=True)
        self._meta: dict[str, str] | None = None
        self._licenses: dict[int, sqlite3.Row] | None = None
        self._licenses_by_key: dict[str, sqlite3.Row] | None = None
        self._aliases: dict[str, tuple[int, str]] | None = None
        self._doc_vectors: tuple[np.ndarray, np.ndarray] | None = None
        self._chunk_vectors: tuple[np.ndarray, np.ndarray] | None = None
        self._chunk_segments: tuple[np.ndarray, np.ndarray] | None = None

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> LicenseStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- metadata ---------------------------------------------------------

    @property
    def meta(self) -> dict[str, str]:
        if self._meta is None:
            self._meta = {row["key"]: row["value"] for row in self.connection.execute("SELECT key, value FROM meta")}
        return self._meta

    def meta_int(self, key: str, default: int = 0) -> int:
        return int(self.meta.get(key, default))

    @property
    def count(self) -> int:
        """Return the number of licenses in the database."""
        return self.meta_int("license_count")

    @property
    def has_vectors(self) -> bool:
        return bool(self.meta_int("embedding_dim"))

    # -- licenses ---------------------------------------------------------

    @property
    def licenses(self) -> dict[int, sqlite3.Row]:
        if self._licenses is None:
            rows = self.connection.execute("SELECT * FROM licenses ORDER BY id")
            self._licenses = {row["id"]: row for row in rows}
        return self._licenses

    def license(self, license_id: int) -> sqlite3.Row:
        return self.licenses[license_id]

    @property
    def licenses_by_key(self) -> dict[str, sqlite3.Row]:
        """Return key -> license row, for callers holding a matched key."""
        if self._licenses_by_key is None:
            self._licenses_by_key = {row["key"]: row for row in self.licenses.values()}
        return self._licenses_by_key

    def text(self, license_id: int) -> tuple[str, str]:
        """Return ``(text, norm_text)`` of a license."""
        row = self.connection.execute(
            "SELECT text, norm_text FROM texts WHERE license_id = ?", (license_id,)
        ).fetchone()
        return row["text"], row["norm_text"]

    # -- shingles ---------------------------------------------------------

    def shingle_pairs(self, license_ids, batch: int = 2000) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return ``license_id -> (shingle set, shingle sequence, bigram set)`` for many licenses.

        Candidate verification needs all three arrays for every shortlisted
        license; fetching them one license at a time would be thousands of
        statements per query, so they are read in batched statements, and all
        three live in one row so a batch is a single table scan. The bigram set is
        precomputed at build time because deriving it per query would dominate
        the cost of scoring a candidate.
        """
        identifiers = list(license_ids)
        pairs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for start in range(0, len(identifiers), batch):
            chunk = identifiers[start : start + batch]
            placeholders = ",".join("?" * len(chunk))
            query = (
                "SELECT license_id, shingle_set, shingle_seq, bigram_set FROM shingle_data "
                f"WHERE license_id IN ({placeholders})"
            )
            for row in self.connection.execute(query, chunk):
                pairs[row["license_id"]] = (
                    np.frombuffer(row["shingle_set"], dtype=np.uint64),
                    np.frombuffer(row["shingle_seq"], dtype=np.uint64),
                    np.frombuffer(row["bigram_set"], dtype=np.uint64),
                )
        return pairs

    def postings(self, shingles: np.ndarray, batch: int = 900):
        """Yield ``(shingle, license_ids)`` for each indexed shingle of ``shingles``.

        ``shingles`` is the query's unique sorted shingle array; batched lookups
        keep the statement below SQLite's variable limit for long queries.
        """
        if shingles.size == 0:
            return
        keys = _as_int64(shingles)
        for start in range(0, keys.size, batch):
            chunk = keys[start : start + batch]
            placeholders = ",".join("?" * chunk.size)
            query = f"SELECT shingle, docs FROM postings WHERE shingle IN ({placeholders})"
            for row in self.connection.execute(query, [int(key) for key in chunk]):
                yield _as_uint64(row["shingle"]), np.frombuffer(row["docs"], dtype=np.int32)

    def digest_index(self) -> dict[str, list[int]]:
        """Return token digest -> license ids, for exact-duplicate reporting."""
        index: dict[str, list[int]] = {}
        for row in self.connection.execute("SELECT id, token_digest FROM licenses ORDER BY id"):
            index.setdefault(row["token_digest"], []).append(row["id"])
        return index

    # -- vectors ----------------------------------------------------------

    def doc_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(license_ids, vectors)`` of the doc level embeddings."""
        if self._doc_vectors is None:
            rows = self.connection.execute("SELECT license_id, vec FROM vectors ORDER BY license_id").fetchall()
            ids = np.array([row["license_id"] for row in rows], dtype=np.int64)
            matrix = np.vstack([np.frombuffer(row["vec"], dtype=np.float32) for row in rows])
            self._doc_vectors = (ids, matrix)
        return self._doc_vectors

    def chunk_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(license_ids, vectors)`` of the chunk level embeddings."""
        if self._chunk_vectors is None:
            rows = self.connection.execute(
                "SELECT license_id, vec FROM chunks ORDER BY license_id, chunk_index"
            ).fetchall()
            ids = np.array([row["license_id"] for row in rows], dtype=np.int64)
            matrix = np.vstack([np.frombuffer(row["vec"], dtype=np.float32) for row in rows])
            self._chunk_vectors = (ids, matrix)
        return self._chunk_vectors

    def chunk_segments(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(license_ids, starts)`` of the contiguous chunk groups.

        Chunks are stored ordered by license, so the per-license maximum over
        chunk similarities is a segmented reduction rather than a scatter.
        """
        if self._chunk_segments is None:
            ids, _ = self.chunk_vectors()
            starts = np.flatnonzero(np.concatenate(([True], ids[1:] != ids[:-1])))
            self._chunk_segments = (ids[starts], starts)
        return self._chunk_segments

    # -- aliases ----------------------------------------------------------

    @property
    def aliases(self) -> dict[str, list[tuple[int, str]]]:
        """Return normalized alias -> ``[(license_id, kind), ...]``.

        A name can legitimately address several licenses (``GNU General Public
        License`` covers every GPL version that carries it), so aliases are
        many-to-many and callers receive every candidate.
        """
        if self._aliases is None:
            aliases: dict[str, list[tuple[int, str]]] = {}
            for row in self.connection.execute("SELECT alias, license_id, kind FROM aliases ORDER BY license_id"):
                aliases.setdefault(row["alias"], []).append((row["license_id"], row["kind"]))
            self._aliases = aliases
        return self._aliases

    def identifier_of(self, license_id: int) -> str:
        """Return the identifier to report: the SPDX key, else the LicenseDB key."""
        row = self.license(license_id)
        return row["spdx_license_key"] or row["key"]

    def info(self) -> dict:
        """Return a JSON-serializable summary of the database."""
        names = [row["name"] for row in self.connection.execute("SELECT name FROM licenses")]
        return {
            "path": str(self.path),
            "meta": self.meta,
            "licenses": len(names),
            "with_spdx_key": sum(
                1
                for row in self.connection.execute(
                    "SELECT spdx_license_key FROM licenses WHERE spdx_license_key IS NOT NULL"
                )
            ),
        }
