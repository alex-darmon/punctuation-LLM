"""Corpus loading: author books, LLM runs, and identity of texts.

Books are identified by content, not by path. `full_books/jane_austen_full.txt`
and `gutenberg_texts/158.txt` are byte-identical punctuation sequences, so they
are one book with two aliases. This matters for leakage: a generation run whose
prompt cited one alias must have the other alias excluded too, and a
path-equality check would miss that.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _rel(path: str | Path) -> str:
    """Repo-relative cache key for an absolute or already-relative path."""
    text = str(path).replace("\\", "/")
    marker = "punctuation-LLM/"
    if marker in text:
        text = text.split(marker)[-1]
    return text.lstrip("./")


def _signature(seq: list[str]) -> str:
    return hashlib.sha256("".join(seq).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Book:
    book_id: str
    author: str
    n_marks: int
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Run:
    """One LLM generation."""

    condition: str
    author: str
    run_id: int
    cache_key: str
    n_marks: int
    source_book_id: str | None


@dataclass
class Corpus:
    cache_path: Path
    cache_sha256: str
    sequences: dict[str, list[str]]
    _books: dict[str, Book]
    _author_books: dict[str, list[str]]
    _alias_to_id: dict[str, str]
    _runs: dict[str, list[Run]] = field(default_factory=dict)

    # ---- human books -----------------------------------------------------
    @property
    def authors(self) -> tuple[str, ...]:
        return tuple(self._author_books)

    def books(self, author: str | None = None) -> list[Book]:
        if author is None:
            return [self._books[b] for a in self._author_books for b in self._author_books[a]]
        return [self._books[b] for b in self._author_books[author]]

    def book_ids(self, author: str) -> list[str]:
        return list(self._author_books[author])

    def marks(self, book_id: str) -> list[str]:
        return self.sequences[book_id]

    def author_of(self, book_id: str) -> str:
        return self._books[book_id].author

    def resolve(self, path_or_key: str | Path) -> str | None:
        """Canonical book_id for a path, alias, or identical-content text."""
        key = _rel(path_or_key)
        if key in self._alias_to_id:
            return self._alias_to_id[key]
        seq = self.sequences.get(key)
        if seq is not None:
            sig = _signature(seq)
            for book_id in self._books:
                if _signature(self.sequences[book_id]) == sig:
                    return book_id
        return None

    def subset(self, authors: list[str]) -> Corpus:
        """A view restricted to the given authors, keeping book identity intact."""
        keep = {a: list(self._author_books[a]) for a in authors}
        ids = {b for bs in keep.values() for b in bs}
        return Corpus(
            cache_path=self.cache_path,
            cache_sha256=self.cache_sha256,
            sequences=self.sequences,
            _books={b: v for b, v in self._books.items() if b in ids},
            _author_books=keep,
            _alias_to_id={k: v for k, v in self._alias_to_id.items() if v in ids},
            _runs=self._runs,
        )

    # ---- LLM runs --------------------------------------------------------
    def load_runs(self, condition: str, directory: str) -> list[Run]:
        """Register the generations of one condition, mapping each to its prompt book."""
        summary_path = ROOT / directory / "all_runs_summary.json"
        source_by_run: dict[tuple[str, int], str] = {}
        if summary_path.exists():
            for row in json.load(open(summary_path)):
                source_by_run[(row["author_key"], int(row["run_id"]))] = row[
                    "source_book_path"
                ]

        runs: list[Run] = []
        for author in self._author_books:
            prefix = f"{directory}/{author}/"
            for key in sorted(k for k in self.sequences if k.startswith(prefix)):
                stem = Path(key).stem
                try:
                    run_id = int(stem.split("_")[-1])
                except ValueError:
                    continue
                src = source_by_run.get((author, run_id))
                runs.append(
                    Run(
                        condition=condition,
                        author=author,
                        run_id=run_id,
                        cache_key=key,
                        n_marks=len(self.sequences[key]),
                        source_book_id=self.resolve(src) if src else None,
                    )
                )
        self._runs[condition] = runs
        return runs

    def runs(self, condition: str, author: str | None = None) -> list[Run]:
        rows = self._runs.get(condition, [])
        if author is None:
            return list(rows)
        return [r for r in rows if r.author == author]


def load_corpus(
    authors_config: str | Path,
    cache: str | Path = "cache/punct_sequences.json",
    exclude_authors: list[str] | None = None,
) -> Corpus:
    """Build a Corpus from a generation-campaign config and the parse cache."""
    cache_path = ROOT / cache if not Path(cache).is_absolute() else Path(cache)
    raw = cache_path.read_bytes()
    sequences = json.loads(raw)["sequences"]
    cfg = json.load(open(ROOT / authors_config if not Path(authors_config).is_absolute() else authors_config))

    drop = set(exclude_authors or [])
    books: dict[str, Book] = {}
    author_books: dict[str, list[str]] = {}
    alias_to_id: dict[str, str] = {}

    for entry in cfg["authors"]:
        author = entry["key"]
        if author in drop:
            continue
        wanted = [_rel(p) for p in (entry.get("book_paths") or [entry["book_path"]])]
        by_sig: dict[str, str] = {}
        ordered: list[str] = []
        for key in wanted:
            seq = sequences.get(key)
            if not seq:
                continue
            sig = _signature(seq)
            if sig in by_sig:
                canonical = by_sig[sig]
                existing = books[canonical]
                books[canonical] = Book(
                    book_id=canonical,
                    author=author,
                    n_marks=existing.n_marks,
                    aliases=existing.aliases + (key,),
                )
                alias_to_id[key] = canonical
                continue
            by_sig[sig] = key
            ordered.append(key)
            books[key] = Book(book_id=key, author=author, n_marks=len(seq))
            alias_to_id[key] = key
        if ordered:
            author_books[author] = ordered

    return Corpus(
        cache_path=cache_path,
        cache_sha256=hashlib.sha256(raw).hexdigest(),
        sequences=sequences,
        _books=books,
        _author_books=author_books,
        _alias_to_id=alias_to_id,
    )
