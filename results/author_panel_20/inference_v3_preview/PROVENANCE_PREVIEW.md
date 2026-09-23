# inference_v3 — PREVIEW RUN, not the record of provenance

Produced 2026-09-19 by running `run_inference_v2.py --config campaigns/inference_v3_repeated_prompt_primary.json`
in Claude's cloud sandbox, because the project venv's interpreter is a macOS rye
symlink that does not resolve inside the Linux bridge, and neither the bridge VM
nor the sandbox can reach PyPI to install the project's pinned dependencies.

**Re-run this on your Mac before citing any number in the paper.** These files carry
no `inference_manifest.json`: the run completed every analysis stage and wrote every
result file, then raised at the final manifest step because
`generated_texts_positional_mechanism_v1/flash_c2/condition_manifest.json` does not
exist (the positional campaign writes `campaign_metadata.json` instead). Either add a
condition manifest for the C2 directories or relax that step before the canonical run.

## Environment substitutions

Two shims stood in for packages that could not be installed. Both are inert here:

- `configargparse` — the vendored `punctuation/config.py` builds its `options` namespace
  through `configargparse.ArgParser`, but `conf/punctuation.ini` has every execution
  setting commented out and each argument carries an explicit `default=`, so the
  namespace is fully determined by `_DEFAULTS`. The shim is `argparse` with
  `is_config_file` stripped. Verified: punctuation_vector, punctuation_end,
  punctuation_quotes, exception_strings and nb_signs all match `_DEFAULTS`.
- `spacy` — imported at module scope by `punctuation_parser` (which also calls
  `spacy.load` at import), but the only function this pipeline takes from that module,
  `get_frequencies`, is pure Python and counts marks in a token list. All sequences come
  from the pre-parsed cache, so nothing is parsed at run time. The stub provides
  `load()` and `lang.en.English` and is never otherwise touched.

numpy 2.4.4 / scipy 1.17.1 (sandbox), not the project pins.

## Validation: inference_v2 reproduced in the same environment

`run_inference_v2.py --config campaigns/inference_v2.json` was run here first and compared
against the committed `results/author_panel_20/inference_v2/`:

| file | result |
|---|---|
| attribution_clustered.csv | byte-identical |
| detection_crossfit.csv | byte-identical |
| detection_folds.csv | byte-identical |
| split_assignments.csv | byte-identical |
| drift_clustered.csv | point estimates and intervals identical; four Wilcoxon p-values differ in the 16th significant figure (scipy version) |
| dash_sensitivity | absent — the raw run texts were not staged; `include_dash_sensitivity_for_raw_runs` is not set in the v3 config |

So the shims and the sandbox numpy/scipy do not move any estimate.

## Cache

`cache/author_panel_20_with_runs_v3.json` was built on your machine by merging
`cache/author_panel_20_with_runs.json` with the 600 C2 entries of
`cache/positional_mechanism_v1.json` — no re-parsing, the sequences already existed.
All 61 human entries were compared entry for entry across the two caches: 0 mismatches.

## Declared reuse checks

| check | result |
|---|---|
| human cache sequences identical to v2 | pass, 61/61 |
| fold assignment identical to v2 | pass, `split_assignments.csv` byte-identical |
| out-of-author thresholds identical to v2 | pass, all 80 threshold rows identical; only `tpr` and `auc` differ, as they must |
