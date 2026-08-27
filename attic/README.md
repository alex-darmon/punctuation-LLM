# Superseded code — do not run

Everything in this directory predates the frozen pipeline and is kept only so the
history of the project stays readable. **No number in the write-up may come from
these scripts.** Use `run_frozen_grid.py` instead.

They are retired rather than trusted because each one pooled author reference
profiles with its own leakage policy, written inline at the point of use. Two of
those policies disagreed, which is how the analysis ended up reporting a
detection AUC of 0.97 and a human attribution baseline of 81%. Both figures came
from scoring text against a reference that still contained the rest of the book
the text was taken from. Under a leave-one-book-out policy the same code paths
give an AUC of 0.78 and a baseline of 47.5%.

The replacement makes the policy impossible to leave implicit:
`punctlib.reference.build_reference_set` has a required `exclude_books` argument,
it resolves exclusions by content so duplicate texts cannot slip through under a
second filename, it raises when an exclusion matches nothing, and it logs every
profile it builds to `results/frozen/reference_audit.csv`.

## What replaced what

| Retired | Replacement |
| --- | --- |
| `tools/replicate_darmon_fig89.py` | `run_frozen_grid.py` → `replication.csv` |
| `tools/check_consistency_ranking.py` | `separability.csv`, `margin_vs_accuracy.csv` |
| `tools/loo_book_attribution.py`, `tools/human_attribution_control.py`, `tools/diagnose_reference_profile.py` | `attribution.csv` (`experiment=human_attribution`) |
| `run_reproducible_kl_analysis.py`, `baseline_analysis.py` | `attribution.csv` (`experiment=llm_attribution`), `dispersion.csv` |
| `tools/detection_test.py` | `detection.csv`, `calibration.csv` |
| `chunk_size_analysis.py` | the `--chunk-sizes` sweep, present in every output table |
| `tools/verify_summary_claims.py` | folded into the grid; it was the one-off audit that found the leak |
| `generate_llm_texts{,_v2,_v3,_v4}.py`, `run_frozen_model_batch.py`, `main.py`, `punctuation_core.py` | `generate_llm_texts_campaign.py`, `punctlib/` |
| `plot_*.py` | none yet; plotting should read the frozen CSVs |

## Legacy outputs

`results-legacy/` holds the CSVs those retired scripts produced. They are the
pre-freeze numbers, including the inflated AUC of 0.97 and the 81% attribution
baseline, kept so the corrections can be traced. The reportable outputs are in
`results/frozen/`.

`tools/build_punct_cache.py` was **not** retired. It is the parsing step that
feeds the grid, and its output is byte-for-byte reproducible: the SHA-256 in
`results/frozen/manifest.json` is checkable against a fresh rebuild.
