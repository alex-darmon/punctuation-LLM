#!/usr/bin/env python3
"""Download and verify the human books declared by an author-panel config.

The panel JSON keeps the experimental selection and its bibliographic metadata
together. Existing local files are never replaced unless ``--refresh`` is
given. Downloads are written atomically and a SHA-256 manifest is emitted so
the exact source corpus used by an extension run can be audited later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
USER_AGENT = "punctuation-stylometry-research/1.0 (Project Gutenberg corpus setup)"


@dataclass(frozen=True)
class BookSpec:
    author_key: str
    author_name: str
    gutenberg_id: int | None
    title: str
    title_fragment: str
    author_fragment: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Author-panel JSON.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Override the source-manifest path declared by the config.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload Gutenberg files even when a local copy exists.",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def load_specs(config_path: Path) -> tuple[dict, list[BookSpec]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    authors = config.get("authors")
    if not isinstance(authors, list) or not authors:
        raise ValueError("panel config must contain a non-empty 'authors' list")

    specs: list[BookSpec] = []
    seen_paths: set[Path] = set()
    for author in authors:
        key = str(author.get("key", ""))
        name = str(author.get("name", ""))
        books = author.get("panel_books")
        if not key or not name or not isinstance(books, list) or not books:
            raise ValueError(
                f"author entry must define key, name, and panel_books: {author}"
            )
        for book in books:
            path = resolve(book["path"])
            if path in seen_paths:
                raise ValueError(f"panel book path is repeated: {path}")
            seen_paths.add(path)
            gutenberg_id = book.get("gutenberg_id")
            specs.append(
                BookSpec(
                    author_key=key,
                    author_name=name,
                    gutenberg_id=(
                        int(gutenberg_id) if gutenberg_id is not None else None
                    ),
                    title=str(book["title"]),
                    title_fragment=str(book.get("title_fragment", book["title"])),
                    author_fragment=str(
                        book.get("author_fragment", author.get("author_fragment", name))
                    ),
                    path=path,
                )
            )
    return config, specs


def candidate_urls(gutenberg_id: int) -> list[str]:
    return [
        f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}.txt",
        f"https://www.gutenberg.org/ebooks/{gutenberg_id}.txt.utf-8",
    ]


def fetch_text(spec: BookSpec, timeout: float, retries: int) -> tuple[str, str]:
    if spec.gutenberg_id is None:
        raise FileNotFoundError(
            f"{spec.path} is local-only and cannot be downloaded automatically"
        )

    failures: list[str] = []
    for url in candidate_urls(spec.gutenberg_id):
        for attempt in range(retries + 1):
            try:
                request = Request(url, headers={"User-Agent": USER_AGENT})
                with urlopen(request, timeout=timeout) as response:
                    payload = response.read()
                return payload.decode("utf-8-sig", errors="replace"), url
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                failures.append(f"{url} (attempt {attempt + 1}): {exc}")
                if attempt < retries:
                    time.sleep(1.0 * (attempt + 1))
    detail = "\n".join(f"  - {line}" for line in failures)
    raise RuntimeError(f"could not download Gutenberg {spec.gutenberg_id}:\n{detail}")


def validate_text(text: str, spec: BookSpec) -> None:
    if len(text) < 10_000:
        raise ValueError(f"{spec.path} is unexpectedly short ({len(text):,} chars)")

    # Bibliographic headers occur before the body. Keeping this bounded avoids
    # accepting a wrong book merely because it mentions the expected author.
    header = " ".join(text[:50_000].casefold().split())
    expected = {
        "title": " ".join(spec.title_fragment.casefold().split()),
        "author": " ".join(spec.author_fragment.casefold().split()),
    }
    for label, fragment in expected.items():
        if fragment not in header:
            raise ValueError(
                f"{spec.path} does not contain expected {label} fragment "
                f"{fragment!r} in its header"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config, specs = load_specs(config_path)
    rows: list[dict] = []

    for index, spec in enumerate(specs, 1):
        downloaded = False
        source_url: str | None = None
        if args.refresh or not spec.path.is_file():
            text, source_url = fetch_text(spec, args.timeout, args.retries)
            validate_text(text, spec)
            write_atomic(spec.path, text)
            downloaded = True
            action = "downloaded"
        else:
            text = spec.path.read_text(encoding="utf-8", errors="replace")
            validate_text(text, spec)
            action = "verified"

        relative = (
            str(spec.path.relative_to(ROOT))
            if spec.path.is_relative_to(ROOT)
            else str(spec.path)
        )
        rows.append(
            {
                "author_key": spec.author_key,
                "author_name": spec.author_name,
                "gutenberg_id": spec.gutenberg_id,
                "title": spec.title,
                "path": relative,
                "source_url": (
                    source_url
                    or (
                        candidate_urls(spec.gutenberg_id)[0]
                        if spec.gutenberg_id is not None
                        else None
                    )
                ),
                "bytes": spec.path.stat().st_size,
                "sha256": sha256(spec.path),
            }
        )
        print(
            f"[{index:02d}/{len(specs):02d}] {action:10s} {relative} "
            f"({spec.path.stat().st_size:,} bytes)"
        )

    manifest_path = resolve(
        args.manifest
        or config.get("source_manifest")
        or "results/author_panel_20/source_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "panel_config": str(config_path.relative_to(ROOT)),
        "panel_config_sha256": sha256(config_path),
        "books": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\n[wrote] {manifest_path.relative_to(ROOT)} ({len(rows)} books)")


if __name__ == "__main__":
    main()
