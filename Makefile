.PHONY: setup setup-ml test lint format app benchmark reproduce smoke paper

setup:                  # tests, lint, benchmark build
	python -m pip install -e ".[dev,research]"

setup-ml:               # models for the app and experiments
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

smoke:                  # quick real run, few subjects/seeds (needs setup-ml; downloads models)
	python -m novavision.experiments.run --backend diffusers --contents 2 --seeds 1 --out results/smoke

reproduce:              # canonical content-track run (needs setup-ml; downloads models)
	python -m novavision.experiments.run --backend diffusers --seeds 3 --out results/paper

paper:                  # regenerate Table 1/2 from the canonical results
	python scripts/report.py --results results/paper/results.json --out paper/tables.md
