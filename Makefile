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

.PHONY: noise
NOISE_SEEDS ?= personaplex-ladder-personaplex-1 personaplex-ladder2-pp-s1
VARIANTS ?= personaplex-ladder2-pp-lr5e-6 personaplex-ladder2-pp-lr1e-4 personaplex-ladder2-pp-r64 personaplex-ladder2-pp-r4
noise:
	PYTHONPATH=src uv run python -m dual.noise runs/ --seeds $(NOISE_SEEDS) --variants $(VARIANTS)

.PHONY: sidebyside
EPOCH ?= 2
sidebyside:
	sh scripts/submit.sh $(NAME) sidebyside --name $(NAME) --base $(BASE) --render $(RENDER) $(if $(RUN),--run $(RUN) --epoch $(EPOCH),) $(ARGS)

.PHONY: position
position:
	PYTHONPATH=src uv run python -m dual.position $(FILES)

.PHONY: engine replay tunnel session
PORT ?= 8998
# ENGINE_RUN, not RUN: measure and sidebyside test `$(if $(RUN),...)` and must keep seeing it unset
ENGINE_RUN ?= cons1-pp-l3
# the engine job: serves until --max-minutes (180) or `sparks abort`; epoch 3 is the job's default
engine:
	sh scripts/submit.sh engine-$(NAME) engine --name engine-$(NAME) --base $(BASE) --run $(ENGINE_RUN) --port $(PORT) $(ARGS)

# the offline check: all 13 holdout dialogues through the engine's own path, no network
replay:
	sh scripts/submit.sh replay-$(NAME) engine --name replay-$(NAME) --base $(BASE) --run $(ENGINE_RUN) --render $(RENDER) --replay 13 $(ARGS)

# the laptop's way in: ENGINE_IP is the container address the engine job prints first
tunnel:
	@test -n "$(ENGINE_IP)" || { echo "ENGINE_IP=<container address from the engine job's log>"; exit 1; }
	@echo "open http://localhost:$(PORT)/ (Chrome, headphones); Ctrl-C closes the tunnel, not the engine"
	ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -L $(PORT):$(ENGINE_IP):$(PORT) $(BOX)

# one recorded session (or the replay) back to runs/ for the record
session:
	mkdir -p runs/$(NAME)
	scp -r '$(BOX):/srv/spark/runs/$(NAME)/engine' runs/$(NAME)/
