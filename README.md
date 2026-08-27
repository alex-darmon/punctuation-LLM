# Punctuation stylometry: human authors versus LLM imitation

Can a statistical test on punctuation features tell human prose from LLM prose,
and can it tell which author an LLM was told to imitate? The features and
distances follow Darmon et al., *Pull out all the stops: Textual analysis via
punctuation sequences* (arXiv:1901.00519), whose figures this pipeline
reproduces as a validity check.

## The pipeline is frozen

Every reported number comes from a single run of `run_frozen_grid.py`, recorded
in `results/frozen/`. Analyses on any other code path are not reportable; the
retired scripts are in `attic/` with an explanation of why.

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
python tools/build_punct_cache.py \
    --run-dir generated_texts_campaign_phaseA_two_samples \
    --run-dir generated_texts_campaign_phaseB_two_samples

python tests/test_reference_builder.py

python run_frozen_grid.py \
    --authors-config campaigns/generation_campaign_phaseA_two_samples.json \
    --condition flash=generated_texts_campaign_phaseA_two_samples \
    --condition pro=generated_texts_campaign_phaseB_two_samples \
    --chunk-sizes 1000 2000 4000 --out results/frozen
```

The parse cache is gitignored but byte-for-byte reproducible; its SHA-256 is
pinned in `results/frozen/manifest.json`, along with the commit, the author list
and the chunk sizes. Two independent runs of the grid produce identical CSVs.

## Layout

| Path | Role |
| --- | --- |
| `punctlib/` | frozen library: corpus, features, the reference builder, statistics |
| `run_frozen_grid.py` | the only analysis entry point; runs all nine experiments |
| `tools/build_punct_cache.py` | parses texts to punctuation sequences (the one input step) |
| `generate_llm_texts_campaign.py` | config-driven generation |
| `campaigns/` | generation configs, which also define the author list |
| `results/frozen/` | the frozen outputs; these are the numbers |
| `tests/` | guardrails for the leakage invariants |
| `attic/` | superseded code, kept for history, not to be run |

## Headline results

From `results/frozen/`, f3 features, 2000-mark chunks, 10 authors, 22 books,
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

**Pro shows positional drift; Flash does not show robust drift.** Every
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

- Em-dashes are converted to commas during cleaning, and Flash's comma share is
  56.5% against 42.9% for humans, so part of that gap is self-inflicted. Results
  on commas are not currently interpretable; the semicolon results are unaffected.
  Fix by saving raw generations and recording dash counts.
- Wilde's two texts include a short-story collection, so his very low
  self-consistency may reflect document type rather than the author.
- Wells has only 14,975 marks across two books, which is thin for a reference.
- Sentence-length features (Mason's request) are not implemented. Pro's full-stop
  share of 46.5% against 25.6% for humans suggests markedly shorter sentences.
