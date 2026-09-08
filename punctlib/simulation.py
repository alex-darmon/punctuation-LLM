"""Process models for the pre-registered punctuation simulation study.

The module deliberately separates text segmentation and parameter fitting from
simulation. A fitted process can therefore be trained on one author fold and
used to generate observations for a disjoint held-out fold.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from spacy.lang.en import English

from punctlib.features import (
    INDEX,
    K,
    PUNCT_VECTOR,
    counts1,
    counts2,
    smooth1,
    smooth2,
)
from punctlib.stats import DF_F1, DF_F3, g_stat_f1, g_stat_f3
from punctlib.text import normalise_text


NARRATION = 0
DIALOGUE = 1
STATE_NAMES = ("narration", "dialogue")

# These are the quote tokens recognised by the frozen parser, plus directional
# single quotes used only to determine state. The latter remain absent from the
# punctuation sequence, matching the existing feature extraction.
EXTRACTED_QUOTE_ALIASES = {"'", "\u201c", "\u201d"}
OPEN_QUOTES = {"\u201c": "\u201d", "\u2018": "\u2019"}
CLOSE_QUOTES = {"\u201d": "\u201c", "\u2019": "\u2018"}
SYMMETRIC_QUOTES = {'"', "'"}
PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")


@dataclass(frozen=True)
class StateBlock:
    state: int
    marks: tuple[str, ...]
    paragraph: int


@dataclass(frozen=True)
class SegmentedText:
    """Punctuation sequence with a deterministic state label for every mark."""

    text: str
    marks: tuple[str, ...]
    states: tuple[int, ...]
    offsets: tuple[int, ...]
    blocks: tuple[StateBlock, ...]
    unmatched_closers: int
    paragraph_unclosed_quotes: int


@dataclass(frozen=True)
class FittedProcess:
    """Parameters estimated from a collection of segmented calibration books."""

    marginal: np.ndarray
    transition: np.ndarray
    state_marginals: tuple[np.ndarray, np.ndarray]
    state_transitions: tuple[np.ndarray, np.ndarray]
    dwell_lengths: tuple[np.ndarray, np.ndarray]
    initial_state: np.ndarray
    n_books: int
    n_marks: int
    state_mark_counts: tuple[int, int]
    state_block_counts: tuple[int, int]


def _paragraph_break_ends(text: str) -> list[int]:
    return [match.end() for match in PARAGRAPH_BREAK.finditer(text)]


def segment_text(
    text: str,
    *,
    strip_gutenberg: bool = True,
    reset_quotes_at_paragraph: bool = True,
) -> SegmentedText:
    """Assign punctuation marks to quote-delimited dialogue or narration.

    Opening and closing quote marks themselves belong to dialogue. Directional
    unmatched closers stay in the current state. Any quote left open at a blank
    line is closed there when ``reset_quotes_at_paragraph`` is true. This makes
    the treatment of multi-paragraph quotations deterministic and prevents one
    damaged quote from relabelling the remainder of a book.
    """
    text = normalise_text(text, strip=strip_gutenberg)
    parser = English(max_length=len(text) + 1)
    tokens = parser(text)
    break_ends = _paragraph_break_ends(text)

    stack: list[str] = []
    marks: list[str] = []
    states: list[int] = []
    offsets: list[int] = []
    paragraphs: list[int] = []
    unmatched_closers = 0
    paragraph_unclosed = 0
    previous_paragraph = 0

    for token in tokens:
        raw = str(token)
        paragraph = bisect.bisect_right(break_ends, token.idx)
        if reset_quotes_at_paragraph and paragraph != previous_paragraph:
            paragraph_unclosed += int(bool(stack))
            stack.clear()
        previous_paragraph = paragraph

        mark_state: int | None = None
        if raw in OPEN_QUOTES:
            stack.append(raw)
            mark_state = DIALOGUE
        elif raw in CLOSE_QUOTES:
            mark_state = DIALOGUE if stack else NARRATION
            expected_open = CLOSE_QUOTES[raw]
            if stack and stack[-1] == expected_open:
                stack.pop()
            elif stack:
                stack.pop()
                unmatched_closers += 1
            else:
                unmatched_closers += 1
        elif raw in SYMMETRIC_QUOTES:
            if stack and stack[-1] == raw:
                mark_state = DIALOGUE
                stack.pop()
            else:
                stack.append(raw)
                mark_state = DIALOGUE

        extracted = raw in PUNCT_VECTOR or raw in EXTRACTED_QUOTE_ALIASES
        if not extracted:
            continue
        mark = '"' if raw in EXTRACTED_QUOTE_ALIASES else raw
        marks.append(mark)
        states.append(mark_state if mark_state is not None else int(bool(stack)))
        offsets.append(int(token.idx))
        paragraphs.append(paragraph)

    paragraph_unclosed += int(bool(stack))

    blocks: list[StateBlock] = []
    block_marks: list[str] = []
    block_state: int | None = None
    block_paragraph: int | None = None
    for mark, state, paragraph in zip(marks, states, paragraphs):
        if block_state is None or state != block_state:
            if block_marks:
                blocks.append(
                    StateBlock(
                        state=int(block_state),
                        marks=tuple(block_marks),
                        paragraph=int(block_paragraph),
                    )
                )
            block_marks = [mark]
            block_state = state
            block_paragraph = paragraph
        else:
            block_marks.append(mark)
    if block_marks:
        blocks.append(
            StateBlock(
                state=int(block_state),
                marks=tuple(block_marks),
                paragraph=int(block_paragraph),
            )
        )

    return SegmentedText(
        text=text,
        marks=tuple(marks),
        states=tuple(states),
        offsets=tuple(offsets),
        blocks=tuple(blocks),
        unmatched_closers=unmatched_closers,
        paragraph_unclosed_quotes=paragraph_unclosed,
    )


def fit_process(
    books: Iterable[SegmentedText],
    *,
    smoothing_eps: float = 0.5,
) -> FittedProcess:
    """Pool calibration books into homogeneous and state-specific parameters."""
    books = list(books)
    if not books:
        raise ValueError("at least one segmented calibration book is required")

    pooled_c1 = np.zeros(K, dtype=float)
    pooled_c2 = np.zeros((K, K), dtype=float)
    state_c1 = [np.zeros(K, dtype=float), np.zeros(K, dtype=float)]
    state_c2 = [np.zeros((K, K), dtype=float), np.zeros((K, K), dtype=float)]
    dwells: list[list[int]] = [[], []]
    initial = np.zeros(2, dtype=float)

    for book in books:
        pooled_c1 += counts1(book.marks)
        pooled_c2 += counts2(book.marks)
        if book.blocks:
            initial[book.blocks[0].state] += 1
        for mark, state in zip(book.marks, book.states):
            state_c1[state][INDEX[mark]] += 1
        # The matrix for state s governs the next mark when the process has
        # entered s, including the first transition at a state boundary.
        for previous, mark, state in zip(
            book.marks, book.marks[1:], book.states[1:]
        ):
            state_c2[state][INDEX[previous], INDEX[mark]] += 1
        for block in book.blocks:
            dwells[block.state].append(len(block.marks))

    if any(not values for values in dwells):
        raise ValueError("both dialogue and narration need at least one block")

    initial = (initial + smoothing_eps) / (initial.sum() + 2 * smoothing_eps)
    return FittedProcess(
        marginal=smooth1(pooled_c1, smoothing_eps),
        transition=smooth2(pooled_c2, smoothing_eps),
        state_marginals=(
            smooth1(state_c1[NARRATION], smoothing_eps),
            smooth1(state_c1[DIALOGUE], smoothing_eps),
        ),
        state_transitions=(
            smooth2(state_c2[NARRATION], smoothing_eps),
            smooth2(state_c2[DIALOGUE], smoothing_eps),
        ),
        dwell_lengths=(
            np.asarray(dwells[NARRATION], dtype=int),
            np.asarray(dwells[DIALOGUE], dtype=int),
        ),
        initial_state=initial,
        n_books=len(books),
        n_marks=int(pooled_c1.sum()),
        state_mark_counts=(
            int(state_c1[NARRATION].sum()),
            int(state_c1[DIALOGUE].sum()),
        ),
        state_block_counts=(len(dwells[NARRATION]), len(dwells[DIALOGUE])),
    )


def _geometric_contrast(
    state_profile: np.ndarray,
    pooled_profile: np.ndarray,
    level: float,
    *,
    axis: int | None = None,
) -> np.ndarray:
    """Scale log-contrast from pooled (0) through empirical (1) and beyond."""
    if level < 0:
        raise ValueError("contrast level must be non-negative")
    log_values = (1.0 - level) * np.log(pooled_profile) + level * np.log(
        state_profile
    )
    values = np.exp(log_values - np.max(log_values, axis=axis, keepdims=True))
    return values / values.sum(axis=axis, keepdims=True)


def state_profiles(
    fitted: FittedProcess,
    contrast: float,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return contrast-adjusted state marginals and transition matrices."""
    marginals = tuple(
        _geometric_contrast(profile, fitted.marginal, contrast)
        for profile in fitted.state_marginals
    )
    transitions = tuple(
        _geometric_contrast(profile, fitted.transition, contrast, axis=1)
        for profile in fitted.state_transitions
    )
    return (
        (marginals[NARRATION], marginals[DIALOGUE]),
        (transitions[NARRATION], transitions[DIALOGUE]),
    )


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """Stationary marginal of a positive row-stochastic transition matrix."""
    values, vectors = np.linalg.eig(np.asarray(transition, dtype=float).T)
    index = int(np.argmin(np.abs(values - 1.0)))
    stationary = np.real(vectors[:, index])
    if stationary.sum() < 0:
        stationary = -stationary
    stationary = np.clip(stationary, 0.0, None)
    return stationary / stationary.sum()


def simulate_iid(
    n_marks: int,
    marginal: np.ndarray,
    rng: np.random.Generator,
) -> list[str]:
    indices = rng.choice(K, size=n_marks, p=marginal)
    return [PUNCT_VECTOR[int(index)] for index in indices]


def simulate_markov(
    n_marks: int,
    transition: np.ndarray,
    initial: np.ndarray,
    rng: np.random.Generator,
) -> list[str]:
    if n_marks <= 0:
        return []
    indices = np.empty(n_marks, dtype=np.int16)
    uniforms = rng.random(n_marks)
    initial_cdf = np.cumsum(initial)
    transition_cdf = np.cumsum(transition, axis=1)
    initial_cdf[-1] = 1.0
    transition_cdf[:, -1] = 1.0
    indices[0] = np.searchsorted(initial_cdf, uniforms[0], side="right")
    for position in range(1, n_marks):
        indices[position] = np.searchsorted(
            transition_cdf[indices[position - 1]],
            uniforms[position],
            side="right",
        )
    return [PUNCT_VECTOR[int(index)] for index in indices]


def simulate_hsmm(
    n_marks: int,
    fitted: FittedProcess,
    *,
    dwell_scale: float,
    contrast: float,
    rng: np.random.Generator,
) -> list[str]:
    """Simulate an alternating two-state hidden semi-Markov punctuation process."""
    if dwell_scale <= 0:
        raise ValueError("dwell_scale must be positive")
    if n_marks <= 0:
        return []

    marginals, transitions = state_profiles(fitted, contrast)
    state = int(rng.choice(2, p=fitted.initial_state))
    transition_cdf = tuple(np.cumsum(matrix, axis=1) for matrix in transitions)
    for matrix in transition_cdf:
        matrix[:, -1] = 1.0

    def sample_dwell(current_state: int) -> int:
        empirical = int(rng.choice(fitted.dwell_lengths[current_state]))
        return max(1, int(np.floor(empirical * dwell_scale + rng.random())))

    indices = np.empty(n_marks, dtype=np.int16)
    indices[0] = rng.choice(K, p=marginals[state])
    remaining = sample_dwell(state) - 1
    uniforms = rng.random(max(n_marks - 1, 0))
    for position in range(1, n_marks):
        if remaining == 0:
            state = DIALOGUE if state == NARRATION else NARRATION
            remaining = sample_dwell(state)
        indices[position] = np.searchsorted(
            transition_cdf[state][indices[position - 1]],
            uniforms[position - 1],
            side="right",
        )
        remaining -= 1
    return [PUNCT_VECTOR[int(index)] for index in indices]


def estimated_profiles(
    sequences: Sequence[Sequence[str]],
    *,
    smoothing_eps: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the same smoothed finite reference used by the real analysis."""
    joined = [mark for sequence in sequences for mark in sequence]
    if not joined:
        raise ValueError("reference sequences contain no punctuation")
    return (
        smooth1(counts1(joined), smoothing_eps),
        smooth2(counts2(joined), smoothing_eps),
    )


def oracle_profiles(
    model: str,
    fitted: FittedProcess,
) -> tuple[np.ndarray, np.ndarray]:
    """Known generating profiles for homogeneous Models 0 and 1."""
    if model == "iid":
        return fitted.marginal, np.tile(fitted.marginal, (K, 1))
    if model == "markov":
        return stationary_distribution(fitted.transition), fitted.transition
    raise ValueError("oracle profiles are defined only for iid and markov models")


def score_sequence(
    sequence: Sequence[str],
    profiles: tuple[np.ndarray, np.ndarray],
) -> dict[str, float]:
    """Return G and raw KL-equivalent divergence in both feature spaces."""
    n = len(sequence)
    g1 = g_stat_f1(counts1(sequence), profiles[0])
    g3 = g_stat_f3(counts2(sequence), profiles[1])
    return {
        "g_f1": g1,
        "g_f3": g3,
        "raw_delta_f1": g1 / (2.0 * n) if n else float("nan"),
        "raw_delta_f3": g3 / (2.0 * max(n - 1, 1)),
        "excess_delta_f1": (
            (g1 - DF_F1) / (2.0 * n) if n else float("nan")
        ),
        "excess_delta_f3": (
            (g3 - DF_F3) / (2.0 * n) if n else float("nan")
        ),
    }
