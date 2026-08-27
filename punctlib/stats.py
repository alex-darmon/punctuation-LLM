"""Distances, test statistics and summaries.

The KL divergence is the vendored implementation, so replication figures stay
comparable with Darmon et al. The likelihood-ratio statistic is the same quantity
rescaled: for the mark-frequency feature, G = 2n * KL(observed || reference),
asymptotically chi-square with k-1 degrees of freedom. That asymptotic reference
is checked empirically rather than assumed, because punctuation marks are not
independent draws.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import mannwhitneyu

from punctlib.features import K
from punctlib.vendored import d_KL

DF_F1 = K - 1
DF_F3 = K * (K - 1)


def kl(p: Sequence[float], q: Sequence[float]) -> float:
    """KL(p || q) as defined by the vendored library."""
    return float(d_KL(list(p), list(q)))


def sym_kl(p: Sequence[float], q: Sequence[float]) -> float:
    """Symmetrised KL, for comparing two samples with no reference/observed roles."""
    return 0.5 * (kl(p, q) + kl(q, p))


def g_stat_f1(observed_counts: np.ndarray, profile: np.ndarray) -> float:
    obs = np.asarray(observed_counts, dtype=float)
    n = obs.sum()
    if n <= 0:
        return float("nan")
    expected = n * np.asarray(profile, dtype=float)
    mask = (obs > 0) & (expected > 0)
    return float(2.0 * np.sum(obs[mask] * np.log(obs[mask] / expected[mask])))


def g_stat_f3(observed_counts: np.ndarray, profile: np.ndarray) -> float:
    """Row-wise likelihood ratio for a first-order chain."""
    obs = np.asarray(observed_counts, dtype=float)
    expected = obs.sum(axis=1, keepdims=True) * np.asarray(profile, dtype=float)
    mask = (obs > 0) & (expected > 0)
    return float(2.0 * np.sum(obs[mask] * np.log(obs[mask] / expected[mask])))


def g_stat(feature: str, observed_counts: np.ndarray, profile: np.ndarray) -> float:
    return g_stat_f1(observed_counts, profile) if feature == "f1" else g_stat_f3(
        observed_counts, profile
    )


def dof(feature: str) -> int:
    return DF_F1 if feature == "f1" else DF_F3


def auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """P(positive > negative), ties at half. NaN if either side is empty."""
    pos = np.asarray(positive, dtype=float)
    neg = np.asarray(negative, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (pos.size * neg.size))


def tpr_at_fpr(positive: Sequence[float], negative: Sequence[float], fpr: float = 0.05) -> tuple[float, float]:
    """Detection rate at a threshold set by the negative (human) distribution."""
    pos = np.asarray(positive, dtype=float)
    neg = np.asarray(negative, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan"), float("nan")
    threshold = float(np.percentile(neg, 100 * (1 - fpr)))
    return float(np.mean(pos > threshold)), threshold
