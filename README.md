# license-rag

[![CI](https://github.com/tardyp/license-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/tardyp/license-rag/actions/workflows/ci.yml)
[![Database](https://github.com/tardyp/license-rag/actions/workflows/database.yml/badge.svg)](https://github.com/tardyp/license-rag/actions/workflows/database.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Arbitrary license text in, ranked SPDX identifiers with a pertinence level out.**

```console
$ license-rag build                            # downloads the latest LicenseDB corpus, builds in ~20s
$ license-rag query LICENSE
1. Apache-2.0  [exact 100%]
   name:    Apache License 2.0
   evidence: the supplied text is this license
   coverage: query passage 100%, query 100%, license 100%, 1512 shingles

$ license-rag query --text "Permission is hereby granted, free of charge, to any person obtaining a copy of this software"
1. MIT-0  [high 82%]
   evidence: 100% of the query is a contiguous passage of this license
   coverage: query passage 100%, query 100%, license 19%, 21 shingles
   also:     MIT, JSON, MIT-STK, ... (same or near-identical text)
```

The whole system is a single SQLite file: 2,715 licenses (~2.7M words) with an inverted shingle
index, precomputed shingle/bigram sets, and precomputed embeddings. Build time ~20 seconds from
a cold cache, query time ~10 ms, no server, no service, no toolkit.

## This is not based on a vector database — and here is why

The engine does use vectors (every license and every 160-token license chunk has an embedding,
and the query path searches them), but **the retrieval is not vector-based, and no vector
database is involved**. It is built on near-duplicate text matching, which is what license
identification actually is. The measured reasons, on this corpus:

| query | lexical channel (shingles) | embedding channel (best of doc/chunk) |
|---|---|---|
| 300-character license header | **85%** top-1 | 60% |
| 400-token excerpt | **90%** | 70% |
| 250-token random excerpt | **91%** | 72% |
| full license text | **100%** | 80–100% |

1. **Embeddings lose on the task.** A license is recognized by which words it contains *in
   which order*, and 5-token shingles capture exactly that; mean-pooled embeddings blur it.
   The embedding channel alone is 15–30 points worse than shingles on every partial-text
   family above.
2. **Their similarity is not separable enough to report a confidence.** On unrelated license
   chunks the cosine sits at 0.57 median and reaches 0.864 at the 99.9th percentile, while an
   excerpt against its own license scores 0.969 at the median. A cosine of 0.9 is inside that
   noise band, and an ANN index narrows the distinction further by approximating.
3. **A pertinence level needs an interpretable quantity.** "94% of your text is a contiguous
   passage of this license" is evidence a reviewer can check and act on. "cosine 0.93" is not,
   and cannot be calibrated into `exact` / `high` / `none` levels with any confidence.
4. **A vector index would buy nothing at this size.** The corpus is 33,662 vectors of 256
   dimensions (34 MB): exact cosine over all of them takes ~5 ms with one matrix multiply, so
   an ANN index would add a dependency, a second artifact and approximate recall in exchange
   for nothing. If the corpus ever grows by two orders of magnitude, swapping in an ANN backend
   is a contained change — `Searcher._semantic_scores` is the only place that touches vectors.

So the design is the inverse of a vector-database RAG: verbatim evidence is primary and decides
the score and the level; embeddings are a secondary channel that can only *lift* a score by the
headroom verbatim matching left unfilled, and a semantic-only match is capped at `medium`. That
is measurable: on lightly reworded license texts, the semantic channel raises the number of
queries rated `medium` from 13/60 to 45/60 while verbatim accuracy is unchanged. See
[`evals/evaluate.py`](evals/evaluate.py) and run `--no-embed` to compare the two modes.

## Install

```console
$ uv tool install git+https://github.com/tardyp/license-rag      # or:
$ pipx install git+https://github.com/tardyp/license-rag
```

Or grab the prebuilt database from the [latest Database
release](https://github.com/tardyp/license-rag/releases/latest) — rebuilt weekly from the newest
corpus — and skip building altogether:

```console
$ curl -L -o data/licensedb.sqlite \
    https://github.com/tardyp/license-rag/releases/latest/download/licensedb.sqlite
$ license-rag query LICENSE --db data/licensedb.sqlite
```

## Build

`build` downloads the corpus from GitHub by itself: it resolves the head commit of
[`aboutcode-org/scancode-licensedb`](https://github.com/aboutcode-org/scancode-licensedb),
downloads that revision's tarball (~14 MB), extracts only the `docs/*.json` license documents,
and caches them under `~/.cache/license-rag` keyed by revision. Repeat builds are offline and
instant, and the exact corpus revision is recorded in the database metadata.

```console
$ license-rag build                                  # latest corpus, ~20s
$ license-rag build --refresh                        # ignore the corpus cache
$ license-rag build --corpus ./docs --db data/db.sqlite   # use a local corpus instead
$ license-rag build --no-embed                       # lexical-only database (no model download)
$ license-rag info                                   # corpus revision, counts, checksums
```

Set `LICENSE_RAG_CACHE` to move the corpus cache, and `GITHUB_TOKEN` to lift GitHub API rate
limits (used only to resolve the revision; the download itself needs no token).

## Query

```console
$ license-rag query LICENSE                 # a file, or - for stdin
$ license-rag query --text "..."            # inline text
$ license-rag query LICENSE --json          # full evidence as JSON
$ license-rag query LICENSE --top-k 10 --min-level high
$ license-rag query LICENSE --fail-under high   # exit 1 unless the best match reaches the level
```

From Python:

```python
from license_rag import Searcher

with Searcher("data/licensedb.sqlite") as searcher:
    for match in searcher.search(license_text).matches:
        print(match.identifier, match.level, match.pertinence, match.reason)
```

`Match` carries the identifier, key, name, category, owner, exception/deprecation flags, the
fused score and level, every evidence component (query passage, query coverage, license
coverage, matched shingles, semantic score), the license passage that matched, and
`also_matches` (licenses with identical or near-identical text).

## Pertinence levels

The level answers "how much do I trust this?" and is graded from the evidence, not from a raw
similarity score.

| level | score | meaning |
|---|---|---|
| `exact` | ≥ 0.97 | the text is this license, its identifier, or a verbatim copy of it |
| `very-high` | ≥ 0.88 | verbatim text, with a small part of the license or of the query missing |
| `high` | ≥ 0.72 | a contiguous passage of the query is in this license, or most of it is |
| `medium` | ≥ 0.50 | partial overlap, reworded text, or a strong semantic match with no verbatim evidence |
| `low` | ≥ 0.30 | weak overlap only |
| `none` | < 0.30 | no usable evidence |

## How it works

Four channels, in order of the strength of their evidence:

1. **Identity** — the query is a license key, an SPDX identifier, a license name, the license
   text itself (by token digest), or carries an `SPDX-License-Identifier` tag. Answered
   directly at full or near-full confidence; a tag scores below a text identity because a tag
   is a declaration *about* a text. Two licenses addressing the same name are both returned.
2. **Lexical candidates** — the query's 5-token shingles are looked up in the inverted index.
   Nothing scans the corpus: candidates are licenses sharing verbatim text.
3. **Verification** — every shortlisted license is scored against the query on three
   independent components: the longest contiguous copy of the query inside the license
   (`query_passage`, measured on consecutive shingle pairs, so order is required), the fraction
   of the query present anywhere in the license (`query_coverage`), and the fraction of the
   license present in the query (`license_coverage`).
4. **Semantics** — static embeddings of the license text and of overlapping 160-token chunks
   (`potion-base-8M`, 256 dimensions, no torch, no GPU). Semantic agreement lifts a lexical
   score only by the *headroom* the lexical channels left unfilled, and semantic-only evidence
   is capped at `medium`.

Ranking keeps the score honest above ordering: matches within 0.02 of a band's best score are
ordered by preference (not deprecated, not an exception, SPDX-listed) rather than by a score
difference the evidence does not support. A fragment of Apache-2.0 is verbatim in ~200 licenses
derived from it, so at that distance the score cannot choose between them — the canonical
license is preferred and the rest are reported as `also_matches`.

## Measured behaviour

[`evals/evaluate.py`](evals/evaluate.py) derives query families from held-out licenses and
reports per-family accuracy plus false-positive checks (`--no-embed` gives the lexical-only
numbers):

| family | n | top-1 | MRR@10 | levels of the top match |
|---|---|---|---|---|
| full text | 60 | 100% | 1.000 | exact 60 |
| reflowed / re-cased | 60 | 100% | 1.000 | exact 60 |
| identifier lookup | 60 | 100% | 1.000 | exact 60 |
| license embedded in a larger file | 33 | 100% | 1.000 | exact 15, very-high 6, high 14 |
| 400-token excerpt | 30 | 90% | 0.950 | high 29, very-high 1 |
| 250-token random excerpt | 33 | 91% | 0.937 | high 28, very-high 5, exact 1 |
| 300-character header | 60 | 85% | 0.886 | high 48, very-high 10, exact 2 |
| lightly reworded text | 60 | 100% | 1.000 | medium 45, low 13, none 2 |

Non-license texts (README prose, source code, a fabricated EULA, prose that merely *mentions* a
license) all score `none` or `medium`, never higher.

The misses are overwhelmingly variant siblings: an excerpt of `lgpl-2.1` returning
`lgpl-2.1-plus`, an excerpt of `opera-eula-2018` returning `opera-eula-eea-2018`. The excerpt
does not contain the clause that distinguishes them, and the engine says so through the level
rather than by guessing. Licenses whose texts are identical or ≥ 95% contained are merged into
one match with the twins in `also_matches`.

## Known limitations

- **Fragments shared across many derived licenses** (an Apache-2.0 definitions passage, a bare
  MIT grant) return a family of equally-scored matches. The canonical license is preferred and
  the others are listed, but the top identifier for such a fragment is a preference, not a
  determination.
- **Generic, textless entries are not in the database.** 21 corpus entries (`unknown`,
  `public-domain`, `proprietary`, `generic-cla`, ...) have no license text and cannot be matched
  by text; the build reports them and records them in `meta.skipped_keys`.
- **A prose mention is not a license text.** "Distributed under the GNU General Public License
  version 2" reaches `medium` through the semantic channel; only a tag or the license text
  itself reaches higher.
- **The level is calibrated on this corpus**, not on legal significance: it measures how much of
  the text was matched, not what the license permits.

## Relation to ScanCode Toolkit

The corpus, the SPDX and LicenseDB keys, and the reference implementation of license detection
all come from [ScanCode Toolkit](https://github.com/aboutcode-org/scancode-toolkit) and the
ScanCode LicenseDB. ScanCode's `licensedcode` engine already detects licenses with a score and a
coverage per match, and it does much more (copyrights, holders, rule-level matches, package
manifests).

This project is not a replacement for it. It packages the narrow question — *which SPDX
identifier is this text, and how much should I trust that?* — as a queryable single-file
database that builds in seconds from the corpus already published on GitHub, with no dependency
on the toolkit, and it makes the confidence a first-class, calibrated output rather than a pair
of internal scores. If you need rule-level evidence, file-level heuristics or copyright
detection, use ScanCode.

## Continuous integration

- [`ci.yml`](.github/workflows/ci.yml) — on every push and pull request: the test suite on
  Python 3.11 and 3.13 (synthetic corpus, no network), `ruff check` and `ruff format --check`,
  then a full build of the database from a freshly downloaded corpus, a smoke query, and the
  evaluation, uploading the resulting database as a build artifact.
- [`database.yml`](.github/workflows/database.yml) — weekly (and on demand): rebuild from the
  newest corpus, evaluate, and publish the database plus its sha256 as a dated release asset.

## Development

```console
$ uv sync --group dev
$ uv run pytest                                          # 30 tests, no network
$ uv run license-rag build                               # build the database
$ uv run python evals/evaluate.py --limit 60             # full evaluation
$ uv run python evals/evaluate.py --limit 60 --no-embed  # lexical channel only
$ uv run ruff check . && uv run ruff format --check .
```

## License

Apache-2.0 (see [LICENSE](LICENSE)). The LicenseDB corpus downloaded at build time is
CC-BY-4.0 from the ScanCode project; the database contains that corpus and inherits its
attribution requirements. See [NOTICE](NOTICE).

No content from ScanCode LicenseDB should be considered or used as legal advice.
