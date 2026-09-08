PYTHON ?= python

FROZEN_CACHE := cache/punct_sequences.json
PANEL_CACHE := cache/author_panel_20_with_runs.json
FLASH_CONDITION := generated_texts_campaign_author20_flash
PRO_CONDITION := generated_texts_campaign_author20_pro
REPRO := results/repro_check

.PHONY: help install check-environment assemble20 cache-frozen cache20 \
	full-grid inference-v2 process-validation process-fit process-simulation process-figures \
	test verify-assets figures tables paper verify reproduce-no-api

help:
	@echo "Supported targets:"
	@echo "  install           Install the pinned Python environment"
	@echo "  assemble20        Assemble frozen-old10 + canonical-new10 runs"
	@echo "  cache20           Rebuild the ignored 20-author parse cache"
	@echo "  full-grid         Regenerate canonical 20-author grid outputs"
	@echo "  inference-v2      Regenerate canonical clustered inference"
	@echo "  process-validation  Create the hand-labelling sample once"
	@echo "  process-fit         Fit the lockable process-parameter artifact"
	@echo "  process-simulation  Run the pre-registered process sweep"
	@echo "  process-figures     Plot completed process-simulation outputs"
	@echo "  test              Run every discovered unit/integration test"
	@echo "  verify            Check canonical artifacts and build manuscript"
	@echo "  reproduce-no-api  Rerun frozen/grid/inference into ignored paths"

install:
	$(PYTHON) -m pip install -r requirements.txt

check-environment:
	$(PYTHON) -c "import sys; assert sys.version_info[:2] == (3, 10), sys.version"
	$(PYTHON) -c "import numpy, scipy, pandas, matplotlib, spacy; import en_core_web_sm"

$(FLASH_CONDITION)/condition_manifest.json:
	$(PYTHON) tools/assemble_generation_condition.py \
		--input generated_texts_campaign_phaseA_two_samples \
		--input generated_texts_campaign_author20_flash_new10 \
		--out $(FLASH_CONDITION)

$(PRO_CONDITION)/condition_manifest.json:
	$(PYTHON) tools/assemble_generation_condition.py \
		--input generated_texts_campaign_phaseB_two_samples \
		--input generated_texts_campaign_author20_pro_new10 \
		--out $(PRO_CONDITION)

assemble20: $(FLASH_CONDITION)/condition_manifest.json $(PRO_CONDITION)/condition_manifest.json

cache-frozen:
	$(PYTHON) tools/build_punct_cache.py \
		--source-manifest campaigns/frozen_cache_sources.json \
		--run-dir generated_texts_campaign_phaseA_two_samples \
		--run-dir generated_texts_campaign_phaseB_two_samples \
		--out $(FROZEN_CACHE)

cache20: assemble20
	$(PYTHON) tools/build_punct_cache.py \
		--authors-config campaigns/author_panel_20.json \
		--run-dir $(FLASH_CONDITION) \
		--run-dir $(PRO_CONDITION) \
		--out $(PANEL_CACHE)

full-grid: cache20
	$(PYTHON) run_frozen_grid.py \
		--authors-config campaigns/author_panel_20.json \
		--cache $(PANEL_CACHE) \
		--condition flash=$(FLASH_CONDITION) \
		--condition pro=$(PRO_CONDITION) \
		--chunk-sizes 1000 2000 4000 \
		--out results/author_panel_20/full_grid

inference-v2: full-grid
	$(PYTHON) run_inference_v2.py

process-validation:
	test -f $(PANEL_CACHE)
	$(PYTHON) run_process_simulation.py --prepare-validation-sample

process-fit:
	test -f $(PANEL_CACHE)
	$(PYTHON) run_process_simulation.py --fit-parameters

process-simulation:
	test -f $(PANEL_CACHE)
	$(PYTHON) run_process_simulation.py

process-figures:
	$(PYTHON) tools/plot_process_simulation.py

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

verify-assets:
	$(PYTHON) tools/verify_reproducibility.py

figures:
	$(PYTHON) tools/plot_inference_v2.py

tables:
	$(PYTHON) tools/render_inference_tables.py

paper: figures tables
	command -v latexmk >/dev/null
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex

verify: check-environment figures tables test verify-assets paper

reproduce-no-api: cache-frozen cache20
	$(PYTHON) run_frozen_grid.py \
		--authors-config campaigns/generation_campaign_phaseA_two_samples.json \
		--cache $(FROZEN_CACHE) \
		--condition flash=generated_texts_campaign_phaseA_two_samples \
		--condition pro=generated_texts_campaign_phaseB_two_samples \
		--chunk-sizes 1000 2000 4000 \
		--out $(REPRO)/frozen
	$(PYTHON) run_frozen_grid.py \
		--authors-config campaigns/author_panel_20.json \
		--cache $(PANEL_CACHE) \
		--condition flash=$(FLASH_CONDITION) \
		--condition pro=$(PRO_CONDITION) \
		--chunk-sizes 1000 2000 4000 \
		--out $(REPRO)/full_grid
	$(PYTHON) run_inference_v2.py --output-dir $(REPRO)/inference_v2
	$(PYTHON) tools/verify_reproducibility.py \
		--candidate-frozen $(REPRO)/frozen \
		--candidate-full-grid $(REPRO)/full_grid \
		--candidate-inference $(REPRO)/inference_v2
