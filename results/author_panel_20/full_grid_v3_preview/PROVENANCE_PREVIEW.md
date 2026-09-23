# full_grid_v3_preview — PREVIEW RUN, not the record of provenance

`run_frozen_grid.py --authors-config campaigns/author_panel_20.json
--cache cache/author_panel_20_with_runs_v3.json
--condition flash=generated_texts_positional_mechanism_v1/flash_c2
--condition pro=generated_texts_positional_mechanism_v1/pro_c2
--out results/author_panel_20/full_grid_v3_preview`

Run 2026-09-21 in the same validated sandbox as inference_v3_preview (see that
directory's PROVENANCE_PREVIEW.md for the environment substitutions and the
byte-for-byte reproduction of inference_v2 that validates them).

## Why this run exists

`tab:leakage-grid` in the paper needs pooled-reference contrasts on the
repeated-prompt runs.  They did not exist: `run_inference_v2.py` does not compute
them, it copies them out of whatever directory `upstream_results.full_grid_dir`
names, and in `campaigns/inference_v3_repeated_prompt_primary.json` that is
`results/author_panel_20/full_grid` -- the SINGLE-PROMPT grid.

**Consequence: `results/author_panel_20/inference_v3/pooled_leakage_contrasts.csv`
is byte-identical to the inference_v2 copy and contains single-prompt numbers.**
It is not a v3 result despite its location.  Verified: both give pooled f3/2,000
TPR 0.9625 (Flash) and 0.8700 (Pro), the old-batch values.

## What the paper uses

`tab:leakage-grid` and the sentence above it are taken from THIS run's
`detection.csv`, f3/2,000: Flash 0.8125 -> 0.9775, Pro 0.3850 -> 0.6350.
The Pro pooled value differs sharply from the single-prompt 0.8700 because the
underlying generated text differs; that is the expected direction, Pro being much
harder to detect under the repeated prompt.

## To make this canonical

1. Re-run the command above on the Mac, writing to `results/author_panel_20/full_grid_v3`.
2. Point `upstream_results.full_grid_dir` in the v3 inference config at that directory
   and re-run the inference, so `pooled_leakage_contrasts.csv` becomes genuinely v3.
3. Re-check `tab:leakage-grid` against the canonical outputs.

Until step 1 is done, the leakage table has no artefact on the Mac.

## Threshold path

This runner estimates the 5% threshold on the grid path, not by cross-fitting, so
its LOBO column differs from `tab:detection` by up to two points.  The same was
true of the v2 grid (Pro 69.5/86.5 against the crossfit 71.5/87.5); the v2 table
happened to agree at the cell the paper quoted.  The caption records this.
