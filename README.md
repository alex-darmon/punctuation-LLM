# Punctuation stylometry: human authors versus LLM imitation

Can a statistical test on punctuation features tell human prose from LLM prose,
and can it tell which author an LLM was told to imitate? The features and
distances follow Darmon et al., *Pull out all the stops: Textual analysis via
punctuation sequences* (arXiv:1901.00519), whose figures this pipeline
reproduces as a validity check.

## Supported evidence layers

The repository deliberately retains two reportable layers:

1. `results/frozen/` is the immutable ten-author discovery/replication analysis.
2. `results/author_panel_20/inference_v2/` is the primary manuscript-facing
   analysis, with author-cluster inference and out-of-author calibration.

The full-corpus D/C extension has its own checksum-verified provenance.
Superseded pilot scripts and the old four-author manuscript are retained in
`attic/`; they are historical evidence, not supported entry points.

The reason for freezing is a leakage bug that survived several rounds of
analysis. Author reference profiles were pooled inline at each point of use, so
each script carried its own exclusion policy, and two of them disagreed. The
detection test scored a human chunk against a profile that still contained the
rest of the book that chunk came from. That produced an AUC of 0.97 and a human
attribution baseline of 81%, both of which fall apart under a leave-one-book-out
policy.

So there is now exactly one way to obtain a profile:

```python
from punctlib import build_reference_set, load_corpus

corpus = load_corpus("campaigns/generation_campaign_phaseA_two_samples.json")
refs = build_reference_set(corpus, exclude_books=[held_out_book])
```

`exclude_books` is required and has no default. A call site that has not decided
what to keep out of the profile has not decided what its experiment measures.
Passing `()` is allowed and means "pool everything", but it has to be written
down, and it shows up as such in the audit log. Alongside that:

- exclusions resolve by **content**, not path, so a text present under two
  filenames cannot re-enter a profile through the second one;
- an exclusion matching nothing **raises**, instead of silently excluding nothing;
- `assert_excluded` is called before scoring, so leakage fails loudly;
- every profile built is written to `results/frozen/reference_audit.csv`.

`tests/test_reference_builder.py` checks each of those properties.

## Reproducing the frozen run

```bash
make install
make cache-frozen
make test
```

The parse cache is gitignored but byte-for-byte reproducible; its SHA-256 is
pinned in `results/frozen/manifest.json`, along with the commit, the author list
and the chunk sizes. `make reproduce-no-api` reruns this grid in
`results/repro_check/frozen/` and byte-compares every CSV with the frozen
outputs; it never overwrites the golden files.

## Full Gutenberg D/C extension

The exact 651-author, 14,947-document corpus used by Darmon et al. is archived
at [Zenodo](https://doi.org/10.5281/zenodo.3605100). It is a 3.23 GB pickle and
is not checked into this repository. Download it into the ignored cache:

```bash
mkdir -p cache/original_paper
curl -L \
  -o cache/original_paper/punctuation_stylometry.p \
  https://zenodo.org/api/records/3605100/files/punctuation_stylometry.p/content
```

Then compute the diagnostic for every author using all of their books:

```bash
python tools/analyze_original_gutenberg_dc.py
```

Here, `C` is the mean directed KL divergence over all ordered pairs of books by
the same author. `D` is the directed KL divergence from that author's pooled
all-book profile to its nearest other author's pooled profile. The script
verifies the archive checksum, reports both the archived rows and an
exact-sequence-deduplicated sensitivity variant, and writes
`results/original_gutenberg_dc/`.

On the exact archive, 10 of 651 author labels have `D/C > 1` for `f1`, and 24
of 651 do so for `f3`. The median ratios are 0.111 and 0.215, respectively.
Removing the two exact within-author duplicate sequences does not change those
counts or medians.

## Balanced 20-author extension

`campaigns/author_panel_20.json` defines the primary extension: the
original ten authors plus ten prose authors screened by the full-corpus `f3`
diagnostic. Every author has exactly three content-distinct books. Holding out
one whole book therefore always leaves a two-book reference.

Prepare and validate the human panel with:

```bash
python tools/prepare_gutenberg_panel.py \
    --config campaigns/author_panel_20.json

python tools/build_punct_cache.py \
    --authors-config campaigns/author_panel_20.json \
    --out cache/author_panel_20_sequences.json

python run_frozen_grid.py \
    --authors-config campaigns/author_panel_20.json \
    --cache cache/author_panel_20_sequences.json \
    --chunk-sizes 1000 2000 4000 \
    --out results/author_panel_20/preflight

python tools/summarize_author_panel_preflight.py
python tests/test_author_panel_20.py
```

The preflight has 20 authors and 60 books. Whole-book leave-one-book-out `f3`
attribution is 80.0% against 5% chance. At 2,000 marks, micro accuracy is 76.7%
and macro-author accuracy is 73.1%. All ten new authors pass the declared
whole-book gate of at least two correct books
out of three. Nine score 3/3 and Robert Sidney Bowen scores 2/3. Luis Senarens
is the weakest short-window case (33.3% at 2,000 marks) despite scoring 3/3 on
whole books, so window-level claims about that author need care.

The generation plans add 200 Flash and 100 Pro runs, using the effective
5,000-mark target recorded by the original campaign outputs:

```bash
python generate_llm_texts_campaign.py \
    --campaign-config campaigns/generation_campaign_new10_flash.json \
    --dry-run

python generate_llm_texts_campaign.py \
    --campaign-config campaigns/generation_campaign_new10_pro.json \
    --dry-run
```

Remove `--dry-run` only when the paid generation should begin. Analysed outputs
retain the old dash-to-comma transformation for comparability, while exact raw
responses are also saved under each author's `raw/` directory.
The resulting new-10 corpora are versioned because model sampling is not
deterministically reproducible. A clean clone therefore needs no credentials
and incurs no API cost. If generation is intentionally repeated, copy
`.env.example` to `.env` and supply an API key, or select ADC. With every run
already present, `--skip-existing` re-analyses and normalises metadata without
initialising a generation backend.

After generation, combine the immutable old runs with the new runs, then build
a separate cache and result grid:

```bash
python tools/assemble_generation_condition.py \
    --input generated_texts_campaign_phaseA_two_samples \
    --input generated_texts_campaign_author20_flash_new10 \
    --out generated_texts_campaign_author20_flash

python tools/assemble_generation_condition.py \
    --input generated_texts_campaign_phaseB_two_samples \
    --input generated_texts_campaign_author20_pro_new10 \
    --out generated_texts_campaign_author20_pro

python tools/build_punct_cache.py \
    --authors-config campaigns/author_panel_20.json \
    --run-dir generated_texts_campaign_author20_flash \
    --run-dir generated_texts_campaign_author20_pro \
    --out cache/author_panel_20_with_runs.json

python run_frozen_grid.py \
    --authors-config campaigns/author_panel_20.json \
    --cache cache/author_panel_20_with_runs.json \
    --condition flash=generated_texts_campaign_author20_flash \
    --condition pro=generated_texts_campaign_author20_pro \
    --chunk-sizes 1000 2000 4000 \
    --out results/author_panel_20/full_grid
```

Reusing the old runs makes this an incremental extension, not a contemporaneous
20-author generation experiment. Any comparison between the old and new author
cohorts may also contain a run-date/provider effect. Regenerate all twenty
authors together if that cohort comparison is itself a target result.

## Evaluation-instrument inference (v2)

`campaigns/inference_v2.json` freezes the primary estimands and sensitivity
analyses for the 20-author study. It declares f3, 2,000 marks, the full panel,
leave-one-book-out references, a 5% detection operating point, author clusters,
five author-stratified folds, and the reproducible bootstrap seed.

Run the additive inference package after the full grid:

```bash
make assemble20
make cache20
make inference-v2
make verify
```

Outputs are written to `results/author_panel_20/inference_v2/`; no file in
`results/frozen/` is changed. The v2 analysis reports author-equal attribution,
paired model contrasts, prompt-source aggregates, five-fold out-of-author
detection calibration, three-book cross-fitted D/C validation, smoothing and
cohort sensitivity, raw dash accounting, and author-blocked positional drift.
`inference_manifest.json` pins the config, cache, generated-text trees, upstream
grid inputs, active source code, complete numerical environment, fold
assignment, bootstrap settings, cluster units, and every analytical output.
For an end-to-end check that does not overwrite committed results or call a
model API, run:

```bash
make reproduce-no-api
```

The manuscript-facing point estimates under the primary specification are
73.1% author-equal human attribution, 8.75% Flash attribution, and 8.0% Pro
attribution, against 5% chance. The Flash--Pro paired difference is not
resolved. Cross-fitted D/C is positively associated with held-out human
accuracy, and the 20-author positional analysis shows target-KL increases for
both models, superseding the frozen 10-author statement that Flash drift was
not detected.

## Layout

| Path | Role |
| --- | --- |
| `punctlib/` | frozen library: corpus, features, the reference builder, statistics |
| `run_frozen_grid.py` | the frozen LLM analysis entry point; runs all nine experiments |
| `run_inference_v2.py` | additive cluster-aware and cross-fitted 20-author inference |
| `campaigns/inference_v2.json` | declared primary estimands, sensitivities, folds and seeds |
| `tools/build_punct_cache.py` | parses texts to punctuation sequences (the one input step) |
| `tools/plot_inference_v2.py` | generates the five principal chapter figures |
| `tools/render_inference_tables.py` | renders manuscript tables from pinned CSVs |
| `tools/verify_reproducibility.py` | verifies source, generation, cache, manifest and golden-output hashes |
| `tools/analyze_original_gutenberg_dc.py` | standalone D/C analysis of the archived full corpus |
| `tools/prepare_gutenberg_panel.py` | downloads, validates and checksum-pins panel books |
| `tools/summarize_author_panel_preflight.py` | enforces the new-author gate and reports balanced accuracy |
| `tools/assemble_generation_condition.py` | combines old and new generations without modifying either |
| `generate_llm_texts_campaign.py` | config-driven generation |
| `campaigns/` | generation configs, which also define the author list |
| `results/frozen/` | the frozen outputs; these are the numbers |
| `results/author_panel_20/` | preflight and later outputs for the 20-author extension |
| `tests/` | guardrails for the leakage invariants |
| `attic/` | superseded code, kept for history, not to be run |
| `Makefile` | supported no-API rebuild, test, verification and paper commands |

## Primary 20-author headline results

The primary specification is fixed in `campaigns/inference_v2.json`: f3,
2,000 punctuation marks, leave-one-book-out target profiles, author-equal
summaries, five out-of-author detection folds, and author-block bootstrap
intervals.

| Result | Estimate |
| --- | --- |
| Human attribution | 73.1% macro (372/485 micro), 5% chance |
| Flash target-author attribution | 8.75% (35/400), 5% chance |
| Pro target-author attribution | 8.0% (16/200), 5% chance |
| Paired Flash minus Pro attribution | +0.75 percentage points, 95% CI −11.8 to +11.3 |
| Flash detection | OOF AUC 0.952; TPR 75.5% at empirical FPR 5.36% |
| Pro detection | OOF AUC 0.931; TPR 71.5% at empirical FPR 5.36% |
| Prospective D/C versus human attribution | Spearman rho 0.534, 95% bootstrap CI 0.021 to 0.895 |
| Flash positional target-KL change, window 1 to 5 | +0.092, 95% CI +0.046 to +0.144 |
| Pro positional target-KL change, window 1 to 5 | +0.168, 95% CI +0.125 to +0.214 |

These are conditional results for this screened panel, these two Gemini models,
and this prompt/generation protocol. Detection here is a punctuation-only
distance instrument, not a universal human-versus-AI classifier.

## Frozen 10-author discovery results

From `results/frozen/`: f3 features, 2,000-mark chunks, 10 authors, 22 books,
200 Gemini 2.5 Flash runs and 100 Gemini 2.5 Pro runs.

| Result | Value |
| --- | --- |
| Replication: between/within-author separation, full documents | 2.09 (f3), 2.16 (f1) |
| Same, references truncated to one middle chunk | 1.64 (f3), 1.56 (f1) |
| Human attribution, leave-one-book-out, 10 authors | 47.5% vs 10% chance |
| Human attribution, margin > 1 subset of 4 authors | 93.0% vs 25% chance |
| LLM attribution, Flash, 10 authors | 15.5%, p = 0.010 |
| LLM attribution, Pro, 10 authors | 6.0%, not above chance |
| LLM attribution on the 4-author subset | Flash 45.0%, Pro 40.0% |
| Detection, leave-one-book-out null | AUC 0.78 (Flash), 0.79 (Pro) |
| Detection, at a 5% false-positive rate | 9.5% and 14.0% |
| Context drift, Flash, first to fifth 1,000-mark window | No significant target drift (f3 ΔKL +0.008, p=0.246) |
| Context drift, Pro, first to fifth 1,000-mark window | Target KL +0.146 (p=0.001); cross-target spread −13.3% |
| Semicolon share, human vs Flash vs Pro | 5.58% / 0.79% / 0.37% |
| Chi-square overdispersion of the human G statistic | 4.6x (f3), 31.9x (f1) |

Four things in this table are worth reading carefully.

**The author subset is selected without looking at accuracy.** `separability.csv`
scores each author by the margin `D/C`, comparing distance to the nearest other
author against how much the author's own books differ from each other. The four
authors with margin > 1 are chosen on human data alone, then reported on. The
margin correlates with per-author accuracy (Spearman 0.747, p = 0.013) but a
threshold at 1 is not clean: Dickens has margin 0.74 and scores 94.1%. It is a
usable screening diagnostic, not a law, and it is close to the quantity the
classifier already thresholds.

**Detection is moderate, not near-ceiling.** The pooled-policy rows in
`detection.csv` reproduce the old AUC of 0.98 and are kept only to show the size
of the leak. The honest figures are around 0.78, and the true-positive rate at a
strict false-positive rate collapses, because genuine human chunks from an unseen
book have a long right tail. Detection does improve with more text: by 4000
marks, Pro reaches AUC 0.856 and a 35% detection rate.

**The chi-square reference fails, and worse than previously thought.**
Overdispersion grows with chunk size (f1: 17.9x, 31.9x, 58.1x at 1000, 2000,
4000 marks), which is what long-range dependence looks like, so no fixed
correction factor will work across chunk sizes. f3 is far better behaved than f1
but still not usable as-is. Build the null empirically from within-author human
comparisons.

**The "AI has less variance" claim is not settled by this data.** Two LLM runs
differ 1.85x as much as two chunks of the same human book, but only 0.75x as much
as two chunks from different books by the same author (`dispersion.csv`). The
sign depends on which human comparison is the right analogue for two independent
generations, so the finding brackets 1.0 rather than pointing one way. The folk
claim is also about sentence length, which these features do not measure.

**The frozen 10-author run finds Pro drift but not robust Flash drift.** Every
generation has at least 5,000 punctuation marks, so `context_drift.csv` compares
five consecutive 1,000-mark windows under the matched leave-one-book-out
reference policy. For Flash, first-to-last changes in target KL, target margin,
rank and prediction concentration are small and not significant at the author
level. For Pro, target KL rises by 0.146 (author-level signed-rank p = 0.001),
the target margin becomes 0.088 more negative (p = 0.014), cross-target
dispersion falls 13.3%, and normalized prediction entropy falls from 0.599 to
0.420. The dominant nearest author shifts from Wells in the first window to
Conrad in the last. This is consistent with Pro moving away from requested
authors and toward a more concentrated positional house style. It does not
establish a causal context-length effect: later windows also occur later in the
narrative and nearer the generated ending. Randomized stopping lengths would be
needed for that causal claim.

## Known issues

- Em-dashes are converted to commas during cleaning. Raw extension-cohort files
  are now retained and `dash_sensitivity.csv` verifies exact dash-to-comma
  accounting; the measured comma-share change is below 0.01 percentage points
  for both models. The older frozen runs lack raw outputs, so this sensitivity
  is limited to the extension cohort.
- Wilde's two texts include a short-story collection, so his very low
  self-consistency may reflect document type rather than the author.
- Wells has only 14,975 marks across two books, which is thin for a reference.
- Sentence-length features (Mason's request) are not implemented. Pro's full-stop
  share of 46.5% against 25.6% for humans suggests markedly shorter sentences.
