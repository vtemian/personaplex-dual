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
	@test -n "$$HF_TOKEN" || { echo "export HF_TOKEN (set -a; . ./.env; set +a)"; exit 1; }
	SECRETS=HF_TOKEN sh scripts/submit.sh fetch-personaplex fetch_weights

probe:
	sh scripts/submit.sh probe-$(NAME) probe --base personaplex --name $(NAME) $(ARGS)

.PHONY: render
render:
	sh scripts/submit.sh render-$(NAME) render --name render-$(NAME) $(ARGS)

.PHONY: train measure bar
BASE ?= personaplex
SLICE ?= 1
RENDER ?= render-1
EPOCHS ?= 8

train:
	sh scripts/submit.sh $(NAME) train --name $(NAME) --base $(BASE) --slice $(SLICE) --render $(RENDER) --epochs $(EPOCHS) $(ARGS)

measure:
	sh scripts/submit.sh measure-$(NAME) measure --name measure-$(NAME) --base $(BASE) --render $(RENDER) $(if $(RUN),--run $(RUN),) $(ARGS)

bar:
	mkdir -p runs
	scp -r '$(BOX):/srv/spark/runs/measure-*/measure' runs/
	@ls runs/measure/*/summary.json >/dev/null 2>&1 || { echo "no runs/measure*/summary.json after scp: run make measure first"; exit 1; }
	PYTHONPATH=src uv run python -m dual.report runs/ > docs/results/bar.md && cat docs/results/bar.md
