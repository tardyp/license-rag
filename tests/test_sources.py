"""Tests for the corpus downloader, using synthetic archives (no network)."""

from __future__ import annotations

import io
import shutil
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from license_rag import sources  # noqa: E402

REVISION = "a" * 40


def make_archive(path: Path) -> Path:
    """Write a corpus tarball shaped like the upstream repository."""
    members = {
        "scancode-licensedb-abc123/docs/mit.json": b'{"key": "mit", "text": "permission is hereby granted"}',
        "scancode-licensedb-abc123/docs/gpl-3.0-only.json": b'{"key": "gpl-3.0-only", "text": "gnu general public"}',
        "scancode-licensedb-abc123/docs/index.json": b"[{...}]",
        "scancode-licensedb-abc123/docs/mit.html": b"<html>not a license document</html>",
        "scancode-licensedb-abc123/docs/mit.yml": b"key: mit",
        "scancode-licensedb-abc123/README.rst": b"license db",
        "scancode-licensedb-abc123/docs/nested/deep.json": b'{"key": "deep"}',
        "../escaped.json": b'{"key": "escaped"}',
    }
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return path


@pytest.fixture()
def archive(tmp_path) -> Path:
    return make_archive(tmp_path / "corpus.tar.gz")


def test_extract_docs_keeps_only_license_documents(archive, tmp_path):
    docs = tmp_path / "out" / "docs"
    written = sources._extract_docs(archive, docs)

    assert written == 2
    assert sorted(path.name for path in docs.iterdir()) == ["gpl-3.0-only.json", "mit.json"]


def test_extract_docs_cannot_escape_the_target(archive, tmp_path):
    out = tmp_path / "out"
    sources._extract_docs(archive, out / "docs")

    assert not (tmp_path / "escaped.json").exists()
    assert not (out.parent / "escaped.json").exists()
    assert not any(path.name == "escaped.json" for path in tmp_path.rglob("escaped.json"))


def test_fetch_corpus_downloads_then_serves_from_cache(archive, tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "latest_revision", lambda *args, **kwargs: REVISION)
    monkeypatch.setattr(sources, "_download", lambda ref, repo, target, progress=None: shutil.copyfile(archive, target))

    first = sources.fetch_corpus(cache_dir=tmp_path, log=lambda *_: None)
    assert first.from_cache is False
    assert first.revision == REVISION
    assert first.label == f"github:{sources.DEFAULT_REPOSITORY}@{REVISION}"
    assert (first.docs_path / "mit.json").is_file()

    def fail(*args, **kwargs):
        raise AssertionError("a cached revision must not be downloaded again")

    monkeypatch.setattr(sources, "_download", fail)
    second = sources.fetch_corpus(cache_dir=tmp_path, log=lambda *_: None)
    assert second.from_cache is True
    assert second.docs_path == first.docs_path


def test_fetch_corpus_refresh_redownloads(archive, tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "latest_revision", lambda *args, **kwargs: REVISION)
    calls = []

    def download(ref, repo, target, progress=None):
        calls.append(ref)
        shutil.copyfile(archive, target)

    monkeypatch.setattr(sources, "_download", download)
    sources.fetch_corpus(cache_dir=tmp_path, log=lambda *_: None)
    refreshed = sources.fetch_corpus(cache_dir=tmp_path, refresh=True, log=lambda *_: None)

    assert len(calls) == 2
    assert refreshed.from_cache is False


def test_fetch_corpus_prunes_older_revisions(archive, tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "latest_revision", lambda *args, **kwargs: "b" * 40)
    monkeypatch.setattr(sources, "_download", lambda ref, repo, target, progress=None: shutil.copyfile(archive, target))
    stale = tmp_path / "corpus-oldrevision"
    stale.mkdir()
    (stale / "docs").mkdir()

    sources.fetch_corpus(cache_dir=tmp_path, log=lambda *_: None)

    assert not stale.exists()
    assert [path.name for path in tmp_path.glob("corpus-*")] == ["corpus-bbbbbbbbbbbb"]


def test_unresolvable_revision_falls_back_to_branch(archive, tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "latest_revision", lambda *args, **kwargs: None)
    refs = []

    def download(ref, repo, target, progress=None):
        refs.append(ref)
        shutil.copyfile(archive, target)

    monkeypatch.setattr(sources, "_download", download)
    source = sources.fetch_corpus(cache_dir=tmp_path, log=lambda *_: None)

    assert refs == [f"refs/heads/{sources.DEFAULT_BRANCH}"]
    assert source.revision is None
    assert source.label == str(source.docs_path)
