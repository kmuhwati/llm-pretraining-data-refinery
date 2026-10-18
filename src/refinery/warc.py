"""Minimal streaming reader for Common Crawl WET files (gzip-compressed WARC).

WET records are `conversion` records holding the plain text Common Crawl
extracted from each HTML page. The reader streams record by record, so a
67 MB file never has to be held in memory.
"""
from __future__ import annotations

import gzip
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WetRecord:
    record_id: str
    url: str
    date: str
    cc_language: str      # Common Crawl's own CLD2 guess, kept for comparison
    text: str


def iter_wet(path: Path) -> Iterator[WetRecord]:
    with gzip.open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith(b"WARC/"):
                continue
            headers = {}
            while (line := f.readline()) not in (b"\r\n", b"\n", b""):
                key, _, value = line.decode("utf-8", "replace").partition(":")
                headers[key.strip().lower()] = value.strip()
            length = int(headers.get("content-length", 0))
            body = f.read(length)
            if headers.get("warc-type") != "conversion":
                continue
            yield WetRecord(
                record_id=headers.get("warc-record-id", "").strip("<>"),
                url=headers.get("warc-target-uri", ""),
                date=headers.get("warc-date", ""),
                cc_language=headers.get("warc-identified-content-language", ""),
                text=body.decode("utf-8", "replace"),
            )
