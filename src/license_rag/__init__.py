"""license-rag: identify the SPDX license of an arbitrary license text.

Build a database from a ScanCode LicenseDB corpus, then query it::

    license-rag build                              # downloads the latest LicenseDB
    license-rag build --corpus ./docs              # or use a local corpus
    license-rag query --text "Permission is hereby granted, free of charge, ..."

Or from Python::

    from license_rag import Searcher

    with Searcher("data/licensedb.sqlite") as searcher:
        for match in searcher.search(text).matches:
            print(match.identifier, match.level, match.pertinence)
"""

from license_rag.build import build_database
from license_rag.embed import DEFAULT_MODEL, Embedder
from license_rag.levels import level_for, pertinence
from license_rag.search import Match, Searcher, SearchResult
from license_rag.sources import CorpusSource, fetch_corpus
from license_rag.store import LicenseStore

__all__ = [
    "DEFAULT_MODEL",
    "CorpusSource",
    "Embedder",
    "LicenseStore",
    "Match",
    "SearchResult",
    "Searcher",
    "build_database",
    "fetch_corpus",
    "level_for",
    "pertinence",
]

__version__ = "0.1.0"
