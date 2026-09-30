"""Getting the license corpus: the latest LicenseDB published on GitHub.

The database is built from the ScanCode LicenseDB corpus, which is published as
a git repository of JSON documents (one per license, under ``docs/``) and
regenerated daily upstream. Rather than vendoring a snapshot, the corpus is
downloaded on demand:

1. resolve the head commit of the upstream branch (one API call, so the build is
   reproducible and cacheable by revision);
2. download that revision's tarball (a single request, ~14 MB);
3. extract only ``*/docs/*.json`` -- the license texts and metadata -- into the
   cache, keyed by revision.

Everything is cached under ``~/.cache/license-rag`` (override with
``LICENSE_RAG_CACHE``), so repeated builds and CI runs are offline after the
first fetch.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# Upstream corpus repository and the branch that tracks its daily regeneration.
DEFAULT_REPOSITORY = "aboutcode-org/scancode-licensedb"
DEFAULT_BRANCH = "main"

# Seconds allowed for each HTTP call.
TIMEOUT = 120
# Reported to the download progress callback, in bytes.
PROGRESS_STEP = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CorpusSource:
    """A resolved corpus: where it came from and where it is on disk."""

    docs_path: Path
    reference: str
    revision: str | None
    from_cache: bool

    @property
    def label(self) -> str:
        """Return a stable, human-readable description of the corpus."""
        if self.revision:
            return f"github:{DEFAULT_REPOSITORY}@{self.revision}"
        return str(self.docs_path)

    def as_dict(self) -> dict:
        return {
            "docs_path": str(self.docs_path),
            "reference": self.reference,
            "revision": self.revision,
            "from_cache": self.from_cache,
        }


def cache_root() -> Path:
    """Return the directory holding downloaded corpora."""
    configured = os.environ.get("LICENSE_RAG_CACHE")
    if configured:
        return Path(configured).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "license-rag"


def _request(url: str, headers: dict[str, str] | None = None) -> urllib.request.addinfourl:
    request = urllib.request.Request(url, headers={"User-Agent": "license-rag", **(headers or {})})
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and "api.github.com" in url:
        request.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(request, timeout=TIMEOUT)  # noqa: S310 - fixed, trusted hosts


def latest_revision(repository: str = DEFAULT_REPOSITORY, branch: str = DEFAULT_BRANCH) -> str | None:
    """Return the head commit sha of ``branch``, or ``None`` if it cannot be resolved.

    Resolving the revision pins a build to an exact corpus revision; when the
    API is unavailable (offline, rate limited), the build falls back to the
    branch tarball or to the cache rather than failing.
    """
    url = f"https://api.github.com/repos/{repository}/commits/{branch}"
    try:
        with _request(url, {"Accept": "application/vnd.github.sha"}) as response:
            revision = response.read().decode("ascii").strip()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None
    return revision if len(revision) == 40 and all(char in "0123456789abcdef" for char in revision) else None


def _download(revision_or_branch: str, repository: str, archive: Path, progress=None) -> None:
    """Download the corpus tarball for ``revision_or_branch`` to ``archive``."""
    url = f"https://codeload.github.com/{repository}/tar.gz/{revision_or_branch}"
    with _request(url) as response, archive.open("wb") as target:
        downloaded = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            target.write(block)
            downloaded += len(block)
            if progress and downloaded // PROGRESS_STEP:
                progress(f"  downloaded {downloaded / 1e6:.1f} MB")
                downloaded %= PROGRESS_STEP


def _extract_docs(archive: Path, docs_path: Path) -> int:
    """Extract the license JSON documents of a corpus tarball; return their count.

    Only ``docs/*.json`` is kept: the HTML and YAML renderings and the duplicate
    ``.LICENSE`` texts of the upstream repository are 4x the bytes for no
    additional information, and members are written explicitly rather than
    through ``extractall``, so no archive path can escape the target directory.
    """
    docs_path.mkdir(parents=True, exist_ok=True)
    written = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if len(parts) < 3 or parts[-2] != "docs" or not parts[-1].endswith(".json"):
                continue
            if parts[-1] == "index.json":  # the corpus index carries no license text
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            with source, (docs_path / parts[-1]).open("wb") as target:
                shutil.copyfileobj(source, target)
            written += 1
    return written


def fetch_corpus(
    repository: str = DEFAULT_REPOSITORY,
    branch: str = DEFAULT_BRANCH,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    log=print,
) -> CorpusSource:
    """Return the corpus documents directory, downloading them if not cached.

    ``refresh`` re-downloads even when the resolved revision is already cached.
    The cache holds one revision per directory and older revisions of both kinds
    are pruned, so a cache stays bounded regardless of how often it is refreshed.
    """
    cache = Path(cache_dir).expanduser() if cache_dir else cache_root()
    cache.mkdir(parents=True, exist_ok=True)

    revision = latest_revision(repository, branch)
    key = revision or branch
    directory = cache / f"corpus-{key[:12]}"
    docs_path = directory / "docs"
    marker = directory / ".complete"

    if not refresh and marker.is_file() and docs_path.is_dir():
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        log(f"corpus: cached {metadata.get('licenses', '?')} licenses from revision {key[:12]} ({directory})")
        return CorpusSource(docs_path, f"{repository}@{branch}", revision, from_cache=True)

    reference = revision or f"refs/heads/{branch}"
    log(f"corpus: downloading {repository}@{key[:12]} from GitHub")
    with tempfile.TemporaryDirectory(dir=cache) as workspace:
        archive = Path(workspace) / "corpus.tar.gz"
        started = time.time()
        _download(reference, repository, archive, progress=log)
        log(f"corpus: downloaded {archive.stat().st_size / 1e6:.1f} MB in {time.time() - started:.1f}s")
        if directory.exists():
            shutil.rmtree(directory)
        count = _extract_docs(archive, docs_path)
        if not count:
            raise RuntimeError(f"no license documents found in the {repository}@{key[:12]} tarball")

    marker.write_text(
        json.dumps(
            {
                "repository": repository,
                "branch": branch,
                "revision": revision,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "licenses": count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for stale in cache.glob("corpus-*"):
        if stale != directory:
            shutil.rmtree(stale, ignore_errors=True)
    log(f"corpus: {count} license documents from revision {key[:12]} ({directory})")
    return CorpusSource(docs_path, f"{repository}@{branch}", revision, from_cache=False)


def main(argv: list[str] | None = None) -> int:
    """Fetch the corpus and print where it is; useful for CI and for debugging."""
    import argparse

    parser = argparse.ArgumentParser(prog="license-rag-corpus", description=__doc__.splitlines()[0])
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--cache", default=None, help="cache directory (default: ~/.cache/license-rag)")
    parser.add_argument("--refresh", action="store_true", help="download even when cached")
    parser.add_argument("--json", action="store_true", help="print the resolved source as JSON")
    args = parser.parse_args(argv)

    source = fetch_corpus(args.repository, args.branch, args.cache, args.refresh)
    print(json.dumps(source.as_dict(), indent=2) if args.json else source.docs_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
