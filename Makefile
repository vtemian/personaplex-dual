CHALKBOARD ?= $(HOME)/projects/ai/bbm-v2
CHALKBOARD_SHA ?= b58c21ac
BOX ?= vlad@spark.local
TAPE ?= /srv/spark/runs/dv3a-tape/tape/speech.jsonl

.PHONY: check lint typecheck test vendor data

check: lint typecheck test

lint:
	uv run ruff format --check src tests
	uv run ruff check src tests

typecheck:
	uv run mypy

test:
	uv run pytest

vendor:
	CHALKBOARD=$(CHALKBOARD) CHALKBOARD_SHA=$(CHALKBOARD_SHA) sh scripts/vendor.sh

data:
	CHALKBOARD=$(CHALKBOARD) BOX=$(BOX) TAPE=$(TAPE) sh scripts/data.sh

.PHONY: smoke fetch-weights probe
NAME ?= dev

smoke:
	sh scripts/submit.sh smoke-$(NAME) smoke $(ARGS)

fetch-weights:
	SECRETS=HF_TOKEN sh scripts/submit.sh fetch-personaplex fetch_weights

probe:
	sh scripts/submit.sh probe-$(NAME) probe --base personaplex $(ARGS)
