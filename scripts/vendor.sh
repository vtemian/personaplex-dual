#!/bin/sh
# scripts/vendor.sh: chalkboard's src/ at one commit, extracted from git, never from the
# working tree, so a render is reproducible whatever bbm-v2 has checked out.
set -eu
: "${CHALKBOARD:?}" "${CHALKBOARD_SHA:?}"
git -C "$CHALKBOARD" rev-parse --verify "$CHALKBOARD_SHA^{commit}" >/dev/null
rm -rf vendor && mkdir -p vendor
git -C "$CHALKBOARD" archive "$CHALKBOARD_SHA" src | tar -x -C vendor --strip-components=1
rm -rf vendor/harness/turns
git -C "$CHALKBOARD" rev-parse "$CHALKBOARD_SHA" > vendor/SHA
PYTHONPATH=vendor uv run python -c "import chalkboard.parser, chalkboard.retime, training.moshi.trunk_lora, harness.engine.audio; print('vendor ok')"
