.PHONY: setup setup-ml test lint format app serve-prod benchmark reproduce text validate-probe \
  validate-probe-scene validate-probe-hf robustness resummarize smoke pilot paper repro-check \
  power ablate-blend submission correct-recovery compare-probes

BIND ?= 127.0.0.1:8000
DIFFUSION_MODEL ?= stabilityai/sd-turbo

setup:                  # tests, lint, benchmark build
	python -m pip install -e ".[dev,research]"

setup-ml:               # model runtime (torch, transformers, diffusers)
	python -m pip install -e ".[ml]"

test:
	pytest -q

lint:
	ruff check novavision tests scripts server.py
	ruff format --check novavision tests scripts server.py

format:
	ruff format novavision tests scripts server.py

app:
	python server.py

# serve-prod keeps --workers 1: the rate limiter, concurrency cap, and model live per process.
serve-prod:             # production WSGI serving (app.run is dev-only); BIND=0.0.0.0:8000 to expose
	python -m gunicorn --workers 1 --threads 4 --timeout 300 --bind $(BIND) server:app

benchmark:              # build full AffectBench from GoEmotions
	python -m novavision.data.build_benchmark --n 100 --out data/affectbench.csv

smoke:                  # quick real run (run: make setup setup-ml; downloads models)
	python -m novavision.experiments.run --backend diffusers --contents 2 --seeds 1 --out results/smoke

pilot:                  # the committed CPU pilot (256-px, 2 subjects, 1 seed -> n=14)
	python -m novavision.experiments.run --backend diffusers --contents 2 --seeds 1 \
	  --width 256 --height 256 --out results/paper

reproduce:              # canonical content-track run, 512-px (needs a GPU); DIFFUSION_MODEL= to swap generator
	python -m novavision.experiments.run --backend diffusers --diffusion-model $(DIFFUSION_MODEL) \
	  --seeds 3 --resume --out results/paper

text:                   # text-conditioned run on AffectBench (run: make benchmark first)
	python -m novavision.experiments.run --backend diffusers --track text \
	  --benchmark data/affectbench.csv --seeds 3 --out results/text

validate-probe:         # probe error on FACES (out-of-domain proxy)
	python -m novavision.eval.validate_probe --hf-dataset FastJobs/Visual_Emotional_Analysis \
	  --n 200 --out results/paper/probe_validation.json

validate-probe-scene:   # probe error IN-DOMAIN on EmoSet scenes (the real ceiling)
	python -m novavision.eval.validate_probe --hf-dataset xodhks/EmoSet118K --label-key emotion \
	  --n 400 --split train --seed 0 --out results/paper/probe_validation_scene.json

validate-probe-hf:      # in-domain error of an INDEPENDENT non-CLIP probe (requires PROBE_MODEL=)
	@test -n "$(PROBE_MODEL)" || { echo "Set PROBE_MODEL=<HF image-classifier model id>"; exit 1; }
	python -m novavision.eval.validate_probe --hf-dataset xodhks/EmoSet118K --label-key emotion \
	  --n 400 --split train --seed 0 --probe hf --probe-model $(PROBE_MODEL) \
	  --out results/paper/probe_validation_scene_hf.json

robustness:             # cross-probe check (requires PROBE_MODEL=<hf image-emotion model id>)
	@test -n "$(PROBE_MODEL)" || { echo "Set PROBE_MODEL=<HF image-classifier model id>"; exit 1; }
	python -m novavision.experiments.run --backend diffusers --contents 4 --seeds 1 \
	  --probe hf --probe-model $(PROBE_MODEL) --out results/robustness

repro-check:            # re-derive the committed headline numbers from the committed records (no models)
	pytest -q tests/test_reproducible_bench.py

resummarize:            # refresh metrics/diagnostics/figures from existing records (no regen)
	python scripts/resummarize.py --results results/paper/results.json

paper:                  # regenerate the script-generated tables from the canonical results
	python scripts/report.py --results results/paper/results.json --out paper/tables.md

power:                  # sample-size analysis for the powered run (no models)
	python scripts/power_analysis.py

correct-recovery:       # probe-error-corrected recovery (Rogan-Gladen; no models)
	python scripts/correct_recovery.py --tier emotion

compare-probes:         # exact McNemar between two probe reports (the paper's p-values; no models)
	python scripts/compare_probes.py results/paper/probe_validation_scene.json results/paper/probe_validation_scene_l14.json

ablate-blend:           # affect-blend sensitivity on the text track (FORCE_C=0|0.8|1; needs models)
	python -m novavision.experiments.run --backend diffusers --track text \
	  --benchmark data/affectbench.csv --seeds 1 --force-coverage $(FORCE_C) \
	  --out results/ablate_c$(FORCE_C)

submission:             # build a schema-valid benchmark submission (SYSTEM="name")
	python scripts/make_submission.py --results results/paper/results.json --system "$(SYSTEM)"
