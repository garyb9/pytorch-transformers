PY ?= python
CARGO ?= cargo
ARGS ?=

.PHONY: install lint format test specs train-tokenizer prepare-data train eval translate export bench rust-build rust-test rust-build-cuda

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check src tests
	$(PY) -m mypy

format:
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

test:
	$(PY) -m pytest

specs:
	@ls specs

train-tokenizer:
	$(PY) -m pytorch_transformers.cli train-tokenizer $(ARGS)

prepare-data:
	$(PY) -m pytorch_transformers.cli prepare-data $(ARGS)

train:
	$(PY) -m pytorch_transformers.cli train $(ARGS)

eval:
	$(PY) -m pytorch_transformers.cli eval $(ARGS)

translate:
	$(PY) -m pytorch_transformers.cli translate $(ARGS)

export:
	$(PY) -m pytorch_transformers.cli export $(ARGS)

bench:
	$(PY) scripts/compare.py $(ARGS)

rust-build:
	cd rust && $(CARGO) build

rust-test:
	cd rust && $(CARGO) test

rust-build-cuda:
	cd rust && $(CARGO) build --features cuda