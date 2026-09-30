"""Holdout evaluation of a built license-rag database.

Query families are derived from licenses held out of the query in ways that
mirror how license text actually reaches a scanner: a whole license file, a
reflowed and re-cased copy, a header comment, an excerpt, a license embedded in
a larger file, a lightly reworded copy, and a bare identifier. Ground truth is
the source license; a match is correct if the source license (or a license
whose normalized text is identical to it) is returned.

Negative texts are included to measure false positives: prose *about* licenses,
source code, a README and a fabricated EULA must not be reported as a confident
license match.

Usage::

    .venv/bin/python evals/evaluate.py --db data/licensedb.sqlite --limit 100
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from license_rag.levels import LEVEL_ORDER, pertinence  # noqa: E402
from license_rag.normalize import normalize_text, tokenize  # noqa: E402
from license_rag.search import Searcher  # noqa: E402
from license_rag.store import LicenseStore  # noqa: E402

SEED = 20260930

# Words substituted in the "paraphrased" family: a crude stand-in for the
# rewording that real-world license variants introduce (vendor names, "License"
# vs "Agreement", reordered clauses).
SYNONYMS = {
    "license": "agreement",
    "software": "program",
    "copyright": "authorship",
    "permission": "authorization",
    "granted": "given",
    "distribute": "deliver",
    "modify": "alter",
    "include": "contain",
    "warranty": "guarantee",
    "liability": "responsibility",
    "terms": "conditions",
    "notice": "statement",
    "must": "shall",
    "may": "can",
    "without": "lacking",
    "source": "origin",
    "code": "implementation",
    "provided": "supplied",
    "users": "recipients",
    "free": "unrestricted",
    "redistribute": "share",
    "use": "employ",
    "licensee": "recipient",
    "licensor": "provider",
}

# Texts that are not a license body, used to detect over-eager matching. Each is
# paired with the highest level that is acceptable for it.
NEGATIVES: list[tuple[str, str, str]] = [
    (
        "about-licenses",
        "medium",
        "The project is released under an open source license approved by the Open Source "
        "Initiative. Contributors agree that their contributions are provided under the same "
        "terms as the project itself, and the maintainers ask that all files carry a short "
        "notice pointing at the LICENSE file at the root of the repository.",
    ),
    (
        "readme",
        "none",
        "Getting started\n\nInstall the package with your favourite package manager, then run "
        "the command line tool. Configuration lives in a TOML file next to the source tree. "
        "Bug reports and pull requests are welcome on the issue tracker; please include the "
        "version you are running and the exact command you executed.",
    ),
    (
        "source-code",
        "none",
        "import os\nfrom pathlib import Path\n\ndef walk(root):\n    for directory, _, files in "
        "os.walk(root):\n        for name in files:\n            if name.endswith('.py'):\n"
        "                yield Path(directory, name)\n\nif __name__ == '__main__':\n    print(len(list(walk('.'))))",
    ),
    (
        "fabricated-eula",
        "low",
        "This document grants you a revocable, non-exclusive right to install one copy of the "
        "accompanying binaries on a single workstation owned by your employer. You may not "
        "reverse engineer the binaries, rent them, or transfer them to a third party without "
        "prior written approval of the vendor, and all support obligations expire thirty days "
        "after activation of the product key.",
    ),
    (
        # Prose that *mentions* a license is not the license text: the semantic
        # channel may recognize the family, but nothing may be claimed beyond
        # medium confidence without verbatim evidence.
        "name-reference",
        "medium",
        "This file is distributed under the terms of the GNU General Public License version 2, "
        "as published by the Free Software Foundation.",
    ),
]

# Texts that name a license through an SPDX tag rather than containing its text.
# Each is paired with the identifiers that must be reported.
NOTICES: list[tuple[str, str, tuple[str, ...]]] = [
    (
        "spdx-tag",
        "/* SPDX-License-Identifier: MIT */\n/* Copyright (c) 2024 Example Inc. */\n",
        ("MIT",),
    ),
    (
        "spdx-tag-expression",
        "#!/bin/sh\n# SPDX-License-Identifier: Apache-2.0 OR MIT\n# Copyright (c) 2024 Example Inc.\n",
        ("Apache-2.0", "MIT"),
    ),
]


def paraphrase(text: str) -> str:
    """Rewrite common license words with synonyms."""
    return " ".join(SYNONYMS.get(token, token) for token in normalize_text(text).split())


def query_families(key: str, text: str, rng: random.Random):
    """Yield ``(family, query_text)`` pairs derived from one license."""
    words = tokenize(normalize_text(text))
    yield "full", text
    yield "reflowed", text.replace("\n", " \n ").upper()
    yield "header-300", text[:300]
    if len(words) >= 400:
        middle = len(words) // 2
        yield "excerpt-400", " ".join(words[middle : middle + 400])
    if len(words) >= 300:
        start = rng.randrange(0, max(1, len(words) - 250))
        yield "excerpt-rand-250", " ".join(words[start : start + 250])
    if len(words) >= 300:
        yield (
            "embedded",
            (
                "#!/usr/bin/env python\n# Copyright (c) 2024 Example Inc.\n\n"
                + text[:4000]
                + "\n\nif __name__ == '__main__':\n    main()\n"
            ),
        )
    yield "paraphrased", paraphrase(text)
    yield "identifier", key


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path(__file__).resolve().parents[1] / "data" / "licensedb.sqlite"))
    parser.add_argument("--limit", type=int, default=100, help="number of licenses to derive queries from")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-embed", action="store_true", help="evaluate the lexical channel alone")
    args = parser.parse_args()

    rng = random.Random(SEED)
    store = LicenseStore(args.db)
    rows = list(store.licenses.values())
    sample = rows if args.limit >= len(rows) else rng.sample(rows, args.limit)

    # Licenses that share a normalized text are interchangeable answers.
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[row["token_digest"]].add(row["key"])

    stats: dict[str, list[float]] = defaultdict(list)
    levels: dict[str, Counter] = defaultdict(Counter)
    started = time.time()
    queries = 0
    with Searcher(args.db, use_embeddings=not args.no_embed) as searcher:
        for row in sample:
            accepted = groups[row["token_digest"]]
            for family, query in query_families(row["key"], store.text(row["id"])[0], rng):
                queries += 1
                result = searcher.search(query, top_k=args.top_k)
                rank = None
                for position, match in enumerate(result.matches, start=1):
                    returned = {match.key, match.identifier, *match.also_matches}
                    if returned & accepted:
                        rank = position
                        break
                stats[family].append(1.0 / rank if rank else 0.0)
                levels[family][result.matches[0].level if result.matches else "none"] += 1

        print(
            f"# {queries} queries over {len(sample)} licenses in {time.time() - started:.1f}s "
            f"({'lexical only' if args.no_embed else 'lexical + semantic'})"
        )
        print(f"{'family':18s} {'n':>5s} {'top1':>7s} {'mrr@10':>7s}  levels (top match)")
        for family, values in sorted(stats.items()):
            top1 = sum(1 for value in values if value == 1.0) / len(values)
            mrr = sum(values) / len(values)
            histogram = " ".join(f"{name}:{count}" for name, count in levels[family].most_common())
            print(f"{family:18s} {len(values):5d} {top1:7.2%} {mrr:7.3f}  {histogram}")

        print("\n# negatives (must not look like a license)")
        failures = 0
        for name, allowed, text in NEGATIVES:
            result = searcher.search(text, top_k=3)
            best = result.matches[0] if result.matches else None
            level = best.level if best else "none"
            # Acceptable means no more pertinent than allowed: a negative must
            # not be reported confidently.
            acceptable = LEVEL_ORDER.index(level) >= LEVEL_ORDER.index(allowed)
            failures += 0 if acceptable else 1
            identifier = best.identifier if best else "-"
            print(
                f"{name:18s} top={identifier:34s} level={level:10s} pertinence="
                f"{pertinence(best.score) if best else 0:3d}%  allowed<={allowed:6s} "
                f"{'ok' if acceptable else 'FAIL'}"
            )

        print("\n# notices (a license named by an SPDX tag or its name)")
        for name, text, expected in NOTICES:
            result = searcher.search(text, top_k=5)
            returned = {
                identifier for match in result.matches for identifier in (match.identifier, *match.also_matches)
            }
            missing = [identifier for identifier in expected if identifier not in returned]
            failures += 1 if missing else 0
            top = result.matches[0] if result.matches else None
            print(
                f"{name:18s} top={(top.identifier if top else '-'):34s} "
                f"level={top.level if top else 'none':10s} expected={','.join(expected)} "
                f"{'ok' if not missing else 'FAIL missing ' + ','.join(missing)}"
            )

    store.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
