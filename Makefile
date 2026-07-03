.PHONY: setup setup-ml test lint format app benchmark reproduce text validate-probe smoke paper repro-check

setup:                  # tests, lint, benchmark build
	python -m pip install -e ".[dev,research]"

setup-ml:               # model runtime (torch, transformers, diffusers)
	python -m pip install -e ".[ml]"

test:
	pytest -q

lint:
	ruff check novavision tests scripts server.py app.py
	ruff format --check novavision tests scripts server.py app.py

format:
	ruff format novavision tests scripts server.py app.py

app:
	python server.py

benchmark:              # build full AffectBench from GoEmotions
	python -m novavision.data.build_benchmark --n 100 --out data/affectbench.csv

smoke:                  # quick real run (run: make setup setup-ml; downloads models)
	python -m novavision.experiments.run --backend diffusers --contents 2 --seeds 1 --out results/smoke

pilot:                  # the committed CPU pilot (256-px, 2 subjects, 1 seed -> n=14)
	python -m novavision.experiments.run --backend diffusers --contents 2 --seeds 1 \
	  --width 256 --height 256 --out results/paper

reproduce:              # canonical content-track run, 512-px (needs a GPU / non-memory-bound box)
	python -m novavision.experiments.run --backend diffusers --seeds 3 --out results/paper

text:                   # text-conditioned run on AffectBench (run: make benchmark first)
	python -m novavision.experiments.run --backend diffusers --track text \
	  --benchmark data/affectbench.csv --seeds 3 --out results/text

validate-probe:         # probe error on FACES (out-of-domain proxy)
	python -m novavision.eval.validate_probe --hf-dataset FastJobs/Visual_Emotional_Analysis \
	  --n 200 --out results/paper/probe_validation.json

validate-probe-scene:   # probe error IN-DOMAIN on EmoSet scenes (the real ceiling)
	python -m novavision.eval.validate_probe --hf-dataset xodhks/EmoSet118K \
	  --n 400 --split train --seed 0 --out results/paper/probe_validation_scene.json

robustness:             # cross-probe check: rerun with an independent non-CLIP probe
	python -m novavision.experiments.run --backend diffusers --contents 4 --seeds 1 \
	  --probe hf --probe-model $(PROBE_MODEL) --out results/robustness

repro-check:            # re-derive the committed headline numbers from the committed records (no models)
	pytest -q tests/test_reproducible_bench.py

resummarize:            # refresh metrics/diagnostics/figures from existing records (no regen)
	python scripts/resummarize.py --results results/paper/results.json

paper:                  # regenerate Table 1/2 from the canonical results
	python scripts/report.py --results results/paper/results.json --out paper/tables.md
