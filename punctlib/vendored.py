"""Single point of contact with the vendored punctuation-stylometry library.

The vendored package reads `sys.argv` at import time to find its .ini config, so
importing it has to happen after the calling script has parsed its own
arguments. Routing every import through this module means that argv rewrite
happens exactly once, in one place, instead of being copy-pasted into each
script (where it has already caused an argument-parsing collision once).

Feature and distance definitions are taken from the vendored library rather than
reimplemented, so replication of Darmon et al. stays faithful to their code.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_VENDOR = ROOT / "punctuation-stylometry-master"
_CONFIG = _VENDOR / "conf" / "punctuation.ini"


def _load():
    if str(_VENDOR) not in sys.path:
        sys.path.insert(0, str(_VENDOR))
    saved = sys.argv
    sys.argv = [saved[0], "-c", str(_CONFIG)]
    try:
        from punctuation.config import options
        from punctuation.feature_operations.distances import d_KL
        from punctuation.feature_operations.matrix_operations import (
            normalised_transition_mat,
            transition_mat,
        )
        from punctuation.parser.punctuation_parser import get_frequencies
    finally:
        sys.argv = saved
    return options, d_KL, normalised_transition_mat, transition_mat, get_frequencies


(
    options,
    d_KL,
    normalised_transition_mat,
    transition_mat,
    get_frequencies,
) = _load()

PUNCT_VECTOR: list[str] = list(options.punctuation_vector)
