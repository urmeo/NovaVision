.PHONY: setup setup-ml test lint format app benchmark reproduce smoke

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

smoke:                  # quick real run on the test fixture (needs setup-ml; downloads models)
	python -m novavision.experiments.run --backend diffusers --benchmark tests/fixtures/affectbench_sample.csv --limit 8 --out results/smoke

reproduce:              # full run on the built benchmark (run `make benchmark` first)
	python -m novavision.experiments.run --backend diffusers --benchmark data/affectbench.csv --out results
