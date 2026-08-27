#!/usr/bin/env python3
"""Guardrail tests for the frozen pipeline.

These check the properties the pipeline is frozen in order to guarantee: that a
reference cannot be built without stating an exclusion policy, that exclusions
follow content rather than paths, and that a silently-ignored exclusion is
impossible.

Run with: python tests/test_reference_builder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from punctlib import (  # noqa: E402
    build_reference,
    build_reference_set,
    load_corpus,
    reset_audit_log,
    audit_log,
)
from punctlib.reference import assert_excluded  # noqa: E402

CONFIG = "campaigns/generation_campaign_phaseA_two_samples.json"
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  pass  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILURES.append(name)


def main() -> None:
    corpus = load_corpus(CONFIG)
    author = "jane_austen"
    print(f"corpus: {len(corpus.authors)} authors, {len(corpus.books())} books\n")

    # 1. exclude_books is required, so no call site can stay silent about leakage.
    try:
        build_reference(corpus, author)  # type: ignore[call-arg]
        check("exclude_books is a required argument", False, "call succeeded")
    except TypeError:
        check("exclude_books is a required argument", True)

    # 2. Duplicate texts are one book, so a two-alias text is not counted twice.
    austen_books = corpus.book_ids(author)
    aliases = [a for b in corpus.books(author) for a in b.aliases]
    check(
        "byte-identical books are merged",
        len(austen_books) == 2 and len(aliases) == 1,
        f"books={austen_books} aliases={aliases}",
    )

    # 3. Excluding by an alias path excludes the canonical copy too. This is the
    #    case that a path-equality check would get wrong.
    alias = aliases[0]
    canonical = corpus.resolve(alias)
    full = build_reference(corpus, author, exclude_books=())
    reduced = build_reference(corpus, author, exclude_books=[alias])
    check(
        "excluding an alias removes the canonical book",
        canonical not in reduced.included_books,
        f"alias={alias} canonical={canonical} included={reduced.included_books}",
    )
    check(
        "exclusion actually shrinks the reference",
        reduced.n_marks < full.n_marks,
        f"{reduced.n_marks} vs {full.n_marks}",
    )

    # 4. An exclusion that matches nothing must raise, never pass silently.
    try:
        build_reference(corpus, author, exclude_books=["does/not/exist.txt"])
        check("unresolvable exclusion raises", False, "call succeeded")
    except ValueError:
        check("unresolvable exclusion raises", True)

    # 5. The leakage assertion fires when a text is inside its own reference.
    try:
        assert_excluded(full, corpus, austen_books[0])
        check("assert_excluded catches leakage", False, "no assertion raised")
    except AssertionError:
        check("assert_excluded catches leakage", True)
    try:
        assert_excluded(reduced, corpus, canonical)
        check("assert_excluded passes a clean reference", True)
    except AssertionError as exc:
        check("assert_excluded passes a clean reference", False, str(exc))

    # 6. A global exclusion applies to whichever author owns the book, and leaves
    #    the other authors untouched.
    refs = build_reference_set(corpus, exclude_books=[austen_books[0]])
    others_intact = all(
        len(refs[a].included_books) == len(corpus.book_ids(a))
        for a in corpus.authors
        if a != author
    )
    check(
        "global exclusion touches only the owning author",
        len(refs[author].included_books) == len(austen_books) - 1 and others_intact,
    )

    # 7. Every build is recorded, so the audit trail cannot miss a profile.
    reset_audit_log()
    _LINE = sys._getframe().f_lineno + 1
    build_reference_set(corpus, exclude_books=())
    check(
        "each profile is logged once per build",
        len(audit_log()) == len(corpus.authors),
        f"{len(audit_log())} entries for {len(corpus.authors)} authors",
    )
    check(
        "audit entries name the caller outside punctlib",
        all(
            not e.caller.startswith("punctlib") and e.caller.endswith(f":{_LINE}")
            for e in audit_log()
        ),
        f"callers={[e.caller for e in audit_log()][:2]}",
    )

    # 8. Excluding every book yields no reference rather than an empty profile.
    single = [a for a in corpus.authors if len(corpus.book_ids(a)) == 2][0]
    check(
        "excluding all books returns None",
        build_reference(corpus, single, exclude_books=corpus.book_ids(single)) is None,
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {', '.join(FAILURES)}")
        sys.exit(1)
    print("all guardrails hold")


if __name__ == "__main__":
    main()
