"""Command line interface: build the database, query it, inspect it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from license_rag import levels
from license_rag.build import build_database
from license_rag.embed import DEFAULT_MODEL
from license_rag.search import Searcher, SearchResult
from license_rag.sources import DEFAULT_BRANCH, DEFAULT_REPOSITORY, fetch_corpus
from license_rag.store import LicenseStore

DEFAULT_DB = "data/licensedb.sqlite"


def _db_path(value: str | None) -> str:
    return value or DEFAULT_DB


def _read_query(source: str | None, text: str | None) -> str:
    if text is not None:
        return text
    if source is None or source == "-":
        return sys.stdin.read()
    return Path(source).read_text(encoding="utf-8", errors="replace")


def _print_result(result: SearchResult, stream=sys.stdout) -> None:
    if result.note:
        print(f"note: {result.note}", file=stream)
    if not result.matches:
        print("no matching license", file=stream)
        return
    for position, match in enumerate(result.matches, start=1):
        flags = "".join(
            flag
            for flag, active in ((", exception", match.is_exception), (", deprecated", match.is_deprecated))
            if active
        )
        print(
            f"{position}. {match.identifier}  [{match.level} {match.pertinence}%]{flags}",
            file=stream,
        )
        if match.name and match.name != match.identifier:
            print(f"   name:    {match.name}", file=stream)
        if match.category:
            print(f"   category: {match.category}", file=stream)
        print(f"   evidence: {match.reason}", file=stream)
        print(
            f"   coverage: query passage {match.lexical.query_passage:.0%}, query "
            f"{match.lexical.query_coverage:.0%}, license "
            f"{match.lexical.license_coverage:.0%}, {match.lexical.matched_shingles} shingles",
            file=stream,
        )
        if match.matched_passage:
            print(f"   passage:  {match.matched_passage[:160]}...", file=stream)
        if match.also_matches:
            print(f"   also:     {', '.join(match.also_matches)} (same or near-identical text)", file=stream)
        if match.is_deprecated and match.replaced_by:
            print(f"   replaced by: {', '.join(match.replaced_by)}", file=stream)


def _command_build(args: argparse.Namespace) -> int:
    log = lambda message: print(message, file=sys.stderr, flush=True)  # noqa: E731
    corpus_source = None
    if args.corpus:
        corpus_path = args.corpus
    else:
        source = fetch_corpus(
            repository=args.repository,
            branch=args.branch,
            cache_dir=args.cache,
            refresh=args.refresh,
            log=log,
        )
        corpus_path = source.docs_path
        corpus_source = source.label
    build_database(
        corpus_path=corpus_path,
        db_path=_db_path(args.db),
        model=args.model,
        with_embeddings=not args.no_embed,
        corpus_source=corpus_source,
        log=log,
    )
    return 0


def _command_query(args: argparse.Namespace) -> int:
    text = _read_query(args.source, args.text)
    if not text.strip():
        print("empty query", file=sys.stderr)
        return 2
    with Searcher(_db_path(args.db), use_embeddings=not args.no_embed) as searcher:
        result = searcher.search(
            text,
            top_k=args.top_k,
            min_level=args.min_level,
            include_duplicates=args.all_duplicates,
        )
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        _print_result(result)
    if args.fail_under:
        best = result.matches[0].level if result.matches else "none"
        return 0 if levels.at_least(best, args.fail_under) else 1
    return 0


def _command_info(args: argparse.Namespace) -> int:
    path = _db_path(args.db)
    with LicenseStore(path) as store:
        info = store.info()
        info["meta"] = store.meta
        info["aliases"] = len(store.aliases)
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the license-rag CLI and return its exit code."""
    parser = argparse.ArgumentParser(
        prog="license-rag",
        description="Identify the SPDX license of arbitrary license text.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build the retrieval database")
    build.add_argument(
        "--corpus",
        default=None,
        help="LicenseDB docs directory or license JSON file (default: download the latest "
        "from github.com/aboutcode-org/scancode-licensedb)",
    )
    build.add_argument("--db", default=None, help=f"database to write (default: {DEFAULT_DB})")
    build.add_argument("--model", default=DEFAULT_MODEL, help="static embedding model")
    build.add_argument("--no-embed", action="store_true", help="skip embeddings (lexical-only database)")
    build.add_argument("--repository", default=DEFAULT_REPOSITORY, help="corpus repository to download")
    build.add_argument("--branch", default=DEFAULT_BRANCH, help="corpus branch to download")
    build.add_argument("--cache", default=None, help="corpus cache directory (default: ~/.cache/license-rag)")
    build.add_argument("--refresh", action="store_true", help="re-download the corpus even when cached")
    build.set_defaults(func=_command_build)

    query = subparsers.add_parser("query", help="identify the license of a text")
    query.add_argument("source", nargs="?", default=None, help="file with the license text, or - for stdin")
    query.add_argument("--text", default=None, help="license text given inline")
    query.add_argument("--db", default=None, help=f"database to read (default: {DEFAULT_DB})")
    query.add_argument("--top-k", type=int, default=5, help="number of matches to report")
    query.add_argument(
        "--min-level", default="none", choices=levels.LEVEL_ORDER, help="only report matches at or above this level"
    )
    query.add_argument("--json", action="store_true", help="print the full result as JSON")
    query.add_argument("--all-duplicates", action="store_true", help="do not merge licenses that share identical texts")
    query.add_argument("--no-embed", action="store_true", help="use the lexical channel only")
    query.add_argument(
        "--fail-under", default=None, choices=levels.LEVEL_ORDER, help="exit 1 unless the best match reaches this level"
    )
    query.set_defaults(func=_command_query)

    info = subparsers.add_parser("info", help="show database metadata")
    info.add_argument("--db", default=None, help=f"database to read (default: {DEFAULT_DB})")
    info.set_defaults(func=_command_info)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
