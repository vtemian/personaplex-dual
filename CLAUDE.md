# personaplex-dual

What this is: `docs/plans/2026-10-26-dual-input-design.md` (the question) and
`docs/plans/2026-10-26-dual-input-batches.md` (the plan, its corrections and decisions).

- `make check` before every commit; work on a branch, never on main.
- GPU work goes through sparks only (`authoring-a-sparks-job`, `operating-the-sparks-queue`).
  Nothing starts by hand on the box.
- `vendor/` is chalkboard's `src/` at `CHALKBOARD_SHA` (`make vendor`); never edit it, never
  commit it. `data/` is fetched by `make data`; never commit it.
- Pure modules in `src/dual/*.py` never import torch at module level except `kl.py`; the
  GPU-only code is `src/dual/model.py` and `src/dual/jobs/*.py`, and imports `moshi` inside
  functions, as chalkboard does.
