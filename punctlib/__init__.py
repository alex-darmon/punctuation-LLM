"""Frozen analysis pipeline for the punctuation LLM study.

Every experiment in this project must obtain its author reference profiles from
`punctlib.reference.build_reference_set`, which requires an explicit
`exclude_books` argument. Nothing else may pool author text into a profile.

The reason is a leakage bug that survived several rounds of analysis: two code
paths (KL attribution and the G-statistic detection test) pooled author text
with different exclusion policies, so the detection result was computed against
a reference that still contained the rest of the book being tested. Inflated
numbers (AUC 0.97, attribution 81%) came from that mismatch, not from the data.

Making the exclusion policy a required argument means a call site cannot stay
silent about it, and `punctlib.reference.audit_log()` records every profile the
run built so the policy can be checked after the fact.
"""

from punctlib.corpus import Corpus, load_corpus
from punctlib.features import (
    FEATURES,
    Features,
    PUNCT_VECTOR,
    chunks,
    counts1,
    counts2,
    features,
)
from punctlib.reference import (
    Reference,
    assert_excluded,
    audit_log,
    build_reference,
    build_reference_set,
    reset_audit_log,
)
from punctlib.stats import auc, g_stat_f1, g_stat_f3, kl, sym_kl

__all__ = [
    "Corpus",
    "FEATURES",
    "Features",
    "PUNCT_VECTOR",
    "Reference",
    "assert_excluded",
    "auc",
    "audit_log",
    "build_reference",
    "build_reference_set",
    "chunks",
    "counts1",
    "counts2",
    "features",
    "g_stat_f1",
    "g_stat_f3",
    "kl",
    "load_corpus",
    "reset_audit_log",
    "sym_kl",
]
