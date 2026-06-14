.PHONY: setup test lint format app benchmark reproduce smoke

setup:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check novavision tests server.py app.py
	ruff format --check novavision tests server.py app.py

format:
	ruff format novavision tests server.py app.py

app:
	python server.py

benchmark:
	python -m novavision.data.build_benchmark --n 100 --out data/affectbench.csv

reproduce:
	python -m novavision.experiments.run --backend diffusers --out results

smoke:
	python -m novavision.experiments.run --backend null --limit 6 --out results/smoke
