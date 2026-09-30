"""Behavioural tests for license-rag.

A small synthetic corpus (no network, no embedding model) is built once per
session and queried through the public API, so these tests exercise the whole
path: normalization, index, scoring, ranking and the CLI.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from license_rag import levels  # noqa: E402
from license_rag.build import build_database  # noqa: E402
from license_rag.cli import main  # noqa: E402
from license_rag.normalize import normalize_text, shingle_set, token_digest  # noqa: E402
from license_rag.scoring import (  # noqa: E402
    bigram_hashes,
    evidence_passage,
    longest_run,
    ordered_shingles,
    score_pair,
)
from license_rag.search import Searcher, spdx_tag_identifiers  # noqa: E402

MIT = (
    "MIT License\n\nCopyright (c) 2024 Example\n\nPermission is hereby granted, free of charge, to "
    "any person obtaining a copy of this software and associated documentation files (the "
    '"Software"), to deal in the Software without restriction, including without limitation the '
    "rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the "
    "Software, and to permit persons to whom the Software is furnished to do so, subject to the "
    "following conditions:\n\nThe above copyright notice and this permission notice shall be "
    "included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "
    '"AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE '
    "WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT."
)

BSD = (
    "BSD 2-Clause License\n\nCopyright (c) 2024, Example\nAll rights reserved.\n\nRedistribution "
    "and use in source and binary forms, with or without modification, are permitted provided that "
    "the following conditions are met:\n\n1. Redistributions of source code must retain the above "
    "copyright notice, this list of conditions and the following disclaimer.\n\n2. Redistributions "
    "in binary form must reproduce the above copyright notice, this list of conditions and the "
    "following disclaimer in the documentation and/or other materials provided with the "
    "distribution.\n\nTHIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS."
)

APACHE = (
    "Apache License\nVersion 2.0, January 2004\nhttp://www.apache.org/licenses/\n\nTERMS AND "
    'CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n\n1. Definitions.\n\n"License" shall '
    "mean the terms and conditions for use, reproduction, and distribution as defined by Sections "
    '1 through 9 of this document.\n\n"Licensor" shall mean the copyright owner or entity '
    "authorized by the copyright owner that is granting the License.\n\n2. Grant of Copyright "
    "License. Subject to the terms and conditions of this License, each Contributor hereby grants "
    "to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright "
    "license to reproduce, prepare Derivative Works of, publicly display, publicly perform, "
    "sublicense, and distribute the Work and such Derivative Works in Source or Object form.\n\n"
    "3. Grant of Patent License. Subject to the terms and conditions of this License, each "
    "Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, "
    "royalty-free, irrevocable patent license to make, have made, use, offer to sell, sell, "
    "import, and otherwise transfer the Work."
)

CORPUS = [
    ("mit", MIT, "MIT", "MIT License"),
    (
        "mit-variant",
        MIT.replace("MIT License\n\nCopyright (c) 2024 Example", "MIT Licence (variant)\n\nCopyright (c) 2024 Example"),
        None,
        "MIT variant",
    ),
    ("bsd-2-clause", BSD, "BSD-2-Clause", "BSD 2-Clause License"),
    ("apache-2.0", APACHE, "Apache-2.0", "Apache License 2.0"),
]


def bigrams_of(text: str) -> np.ndarray:
    """Return the query-side bigram sequence: order matters, duplicates kept."""
    return bigram_hashes(ordered_shingles(text))


MIT_BIGRAMS = np.unique(bigram_hashes(ordered_shingles(MIT)))


UNRELATED = (
    "Getting started\n\nInstall the package with your package manager, then run the command line "
    "tool. Configuration lives in a TOML file next to the source tree. Bug reports and pull "
    "requests are welcome on the issue tracker."
)


@pytest.fixture(scope="session")
def db(tmp_path_factory) -> Path:
    """Build an embeddings-free database over the synthetic corpus."""
    corpus = tmp_path_factory.mktemp("corpus")
    for key, text, spdx, name in CORPUS:
        (corpus / f"{key}.json").write_text(
            json.dumps({"key": key, "text": text, "spdx_license_key": spdx, "name": name}),
            encoding="utf-8",
        )
    path = corpus / "licensedb.sqlite"
    build_database(corpus, path, with_embeddings=False, log=lambda *_: None)
    return path


@pytest.fixture(scope="session")
def searcher(db):
    with Searcher(db, use_embeddings=False) as instance:
        yield instance


# -- normalization --------------------------------------------------------


def test_token_digest_ignores_formatting_only():
    assert token_digest(MIT) == token_digest(MIT.upper())
    assert token_digest(MIT) == token_digest(MIT.replace("\n", "   "))
    assert token_digest(MIT) == token_digest(MIT.replace(",", " , "))
    # A changed word is a different license text.
    assert token_digest(MIT) != token_digest(MIT.replace("permission", "permissions"))


def test_shingles_ignore_case_and_reflow():
    import numpy as np

    assert np.array_equal(shingle_set(MIT), shingle_set(MIT.upper().replace("\n", " ")))


# -- scoring --------------------------------------------------------------


def test_verbatim_copy_is_a_full_passage():
    query = normalize_text(MIT)
    score = score_pair(shingle_set(query), bigrams_of(query), MIT_BIGRAMS, shingle_set(MIT))
    assert score.query_passage == 1.0
    assert score.query_coverage == 1.0
    assert score.license_coverage == 1.0
    assert score.similarity == pytest.approx(1.0)


def test_snippet_is_a_passage_but_not_the_whole_license():
    query = " ".join(normalize_text(MIT).split()[:40])
    score = score_pair(shingle_set(query), bigrams_of(query), MIT_BIGRAMS, shingle_set(MIT))
    assert score.query_passage == 1.0
    assert score.license_coverage < 0.5


def test_reordered_blocks_lose_passage_but_keep_coverage():
    """Reordering whole passages keeps the wording (coverage) but loses the copy (passage)."""
    paragraphs = [part for part in MIT.split("\n\n") if part.strip()]
    query = "\n\n".join(reversed(paragraphs))
    assert normalize_text(query) != normalize_text(MIT)
    reordered = score_pair(shingle_set(query), bigrams_of(query), MIT_BIGRAMS, shingle_set(MIT))
    verbatim = score_pair(shingle_set(normalize_text(MIT)), bigrams_of(MIT), MIT_BIGRAMS, shingle_set(MIT))
    assert reordered.query_coverage > 0.85
    assert reordered.query_passage < 0.9
    assert reordered.similarity < verbatim.similarity


def test_word_shuffled_text_matches_nothing():
    """Shuffling words destroys every 5-gram: this is not the same license text at all."""
    words = normalize_text(MIT).split()
    random.Random(7).shuffle(words)
    query = " ".join(words)
    shuffled = score_pair(shingle_set(query), bigrams_of(query), MIT_BIGRAMS, shingle_set(MIT))
    assert shuffled.matched_shingles == 0
    assert shuffled.similarity == 0.0


def test_passage_never_exceeds_one():
    query = normalize_text(MIT) + " " + normalize_text(MIT)  # repeated phrases
    score = score_pair(shingle_set(query), bigrams_of(query), MIT_BIGRAMS, shingle_set(MIT))
    assert 0.0 <= score.query_passage <= 1.0
    assert 0.0 <= score.query_coverage <= 1.0
    assert 0.0 <= score.license_coverage <= 1.0


def test_longest_run_and_evidence_passage():
    import numpy as np

    assert longest_run(np.array([False, True, True, False, True])) == (2, 1)
    assert longest_run(np.array([False, False])) == (0, -1)
    start, length = evidence_passage(shingle_set(normalize_text(MIT)), ordered_shingles(MIT))
    assert start >= 0 and length > 0


# -- levels ---------------------------------------------------------------


def test_levels_are_ordered_and_monotone():
    scores = [1.0, 0.9, 0.8, 0.6, 0.4, 0.1]
    names = [levels.level_for(score) for score in scores]
    assert names == ["exact", "very-high", "high", "medium", "low", "none"]
    assert all(levels.at_least(names[i], names[i + 1]) for i in range(len(names) - 1))


def test_semantic_alone_cannot_claim_high_confidence():
    assert levels.level_for(levels.fuse(0.0, 1.0)) == "medium"
    assert levels.fuse(0.0, 1.0) < levels.SEMANTIC_ONLY_CAP + 1e-9


def test_semantic_does_not_reorder_saturated_lexical_evidence():
    assert levels.fuse(0.75, 1.0, headroom=0.0) == pytest.approx(0.75)


# -- end to end -----------------------------------------------------------


def test_full_text_is_identified_exactly(searcher):
    match = searcher.search(MIT).matches[0]
    assert match.identifier == "MIT"
    assert match.level == "exact"
    assert match.pertinence == 100


def test_reflowed_and_recased_text_is_still_exact(searcher):
    match = searcher.search(MIT.upper().replace("\n", " \n ")).matches[0]
    assert match.identifier == "MIT"
    assert match.level == "exact"


def test_header_snippet_finds_the_license(searcher):
    result = searcher.search(" ".join(MIT.split()[:40]))
    assert result.matches
    top = result.matches[0]
    assert top.identifier in {"MIT", *top.also_matches}
    assert levels.at_least(top.level, "high")


def test_middle_excerpt_finds_the_license(searcher):
    words = APACHE.split()
    match = searcher.search(" ".join(words[40:110])).matches[0]
    assert match.identifier == "Apache-2.0"
    assert match.lexical.query_passage > 0.9


def test_unrelated_text_is_not_a_license(searcher):
    result = searcher.search(UNRELATED)
    assert not result.matches or result.matches[0].level == "none"


def test_identical_texts_are_reported_as_one_match(searcher):
    match = searcher.search(MIT).matches[0]
    assert match.also_matches, "the twin license must be reported alongside"
    assert "mit-variant" in match.also_matches


def test_identifier_lookup(searcher):
    match = searcher.search("Apache-2.0").matches[0]
    assert match.identifier == "Apache-2.0"
    assert match.level == "exact"
    assert "identifier" in match.reason


def test_spdx_tag_is_resolved(searcher):
    result = searcher.search("/* SPDX-License-Identifier: Apache-2.0 */\n/* Copyright (c) 2024 */")
    assert result.matches[0].identifier == "Apache-2.0"
    assert "SPDX-License-Identifier" in result.matches[0].reason
    assert spdx_tag_identifiers("# SPDX-License-Identifier: MIT OR Apache-2.0") == ["MIT", "Apache-2.0"]


def test_short_query_reports_no_shingles(searcher):
    result = searcher.search("hello world")
    assert result.query_shingles == 0
    assert result.note


def test_result_serializes_to_json(searcher):
    payload = json.dumps(searcher.search(MIT).as_dict())
    assert json.loads(payload)["matches"][0]["identifier"] == "MIT"


# -- CLI ------------------------------------------------------------------


def test_cli_query_json(db, capsys):
    assert main(["query", "--db", str(db), "--text", MIT, "--json", "--no-embed"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["matches"][0]["identifier"] == "MIT"
    assert payload["matches"][0]["level"] == "exact"


def test_cli_fail_under_sets_exit_code(db):
    assert main(["query", "--db", str(db), "--text", UNRELATED, "--no-embed", "--fail-under", "high"]) == 1
    assert main(["query", "--db", str(db), "--text", MIT, "--no-embed", "--fail-under", "high"]) == 0


def test_cli_info(db, capsys):
    assert main(["info", "--db", str(db)]) == 0
    assert json.loads(capsys.readouterr().out)["licenses"] == len(CORPUS)
