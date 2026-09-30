"""Loading of the ScanCode LicenseDB corpus.

The corpus is the ``docs/`` directory of the LicenseDB repository: one
``<key>.json`` document per license, each carrying the license metadata plus the
full ``text`` of the license. ``index.json`` is the corpus index and is skipped
here because it has no text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LicenseRecord:
    """One license of the corpus."""

    key: str
    text: str
    name: str = ""
    short_name: str = ""
    spdx_license_key: str | None = None
    other_spdx_license_keys: tuple[str, ...] = ()
    category: str = ""
    owner: str = ""
    homepage_url: str = ""
    is_exception: bool = False
    is_deprecated: bool = False
    replaced_by: tuple[str, ...] = ()
    minimum_coverage: int | None = None

    @property
    def identifier(self) -> str:
        """Return the identifier to report for this license.

        The SPDX key when the license has one, otherwise the LicenseDB key:
        both are SPDX license expression identifiers, and LicenseDB keys are the
        ones ScanCode itself reports for the 132 licenses SPDX does not cover.
        """
        return self.spdx_license_key or self.key


def _record_from_json(data: dict) -> LicenseRecord | None:
    text = data.get("text")
    if not text or not text.strip():
        return None
    key = data.get("key")
    if not key:
        return None

    def _tuple(value) -> tuple[str, ...]:
        if not value:
            return ()
        if isinstance(value, str):
            return (value,)
        return tuple(value)

    return LicenseRecord(
        key=key,
        text=text,
        name=data.get("name") or "",
        short_name=data.get("short_name") or "",
        spdx_license_key=data.get("spdx_license_key") or None,
        other_spdx_license_keys=_tuple(data.get("other_spdx_license_keys")),
        category=data.get("category") or "",
        owner=data.get("owner") or "",
        homepage_url=data.get("homepage_url") or "",
        is_exception=bool(data.get("is_exception")),
        is_deprecated=bool(data.get("is_deprecated")),
        replaced_by=_tuple(data.get("replaced_by")),
        minimum_coverage=data.get("minimum_coverage"),
    )


def load_corpus(path: str | Path) -> tuple[list[LicenseRecord], list[str]]:
    """Load every license under ``path``.

    ``path`` is either a LicenseDB ``docs`` directory or a single license JSON
    file. Returns ``(records, skipped_keys)``; keys skipped for having no text
    are reported so the build never silently drops licenses.
    """
    path = Path(path)
    files = [path] if path.is_file() else sorted(p for p in path.glob("*.json") if p.name != "index.json")

    records: list[LicenseRecord] = []
    skipped: list[str] = []
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        if isinstance(data, list):  # the corpus index
            continue
        record = _record_from_json(data)
        if record is None:
            skipped.append(file.stem)
        else:
            records.append(record)
    return records, skipped
