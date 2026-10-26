#!/bin/sh
# scripts/submit.sh NAME JOB [args...]: assemble the context, submit, print the job id.
# JOB is a module under dual.jobs. $SECRETS (space-separated names) become --secret flags.
set -eu
name="$1"; job="$2"; shift 2
[ -f vendor/SHA ] || { echo "run make vendor first"; exit 1; }
if [ -z "${IMAGE:-}" ] && [ -z "${DUAL_ALLOW_DIRTY:-}" ] && [ -n "$(git status --porcelain -- src)" ]; then
  echo "uncommitted changes under src/: the image would carry the previous commit's tag"; exit 1
fi
# -L: vendor/ (and src/, in a worktree) may be symlinks; Docker cannot follow a link
# out of its context, so the context must hold the real files.
rm -rf docker/src docker/vendor && cp -RL src docker/src && cp -RL vendor docker/vendor
set -- python -m "dual.jobs.$job" "$@"
secrets=""; for s in ${SECRETS:-}; do secrets="$secrets --secret $s"; done
if [ -n "${IMAGE:-}" ]; then
  sparks submit --image "$IMAGE" --data "${DATA:-data}" --name "$name" $secrets -- "$@"
else
  sparks submit --context docker --data "${DATA:-data}" --name "$name" $secrets -- "$@"
fi
