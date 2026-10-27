#!/bin/sh
# Eight runs, one queue: both bases over the four rungs of render-1, then one measure each.
# Runs are submitted one at a time, so a box hang loses at most one; FROM=<run name> skips
# the runs before it (a hung train is resumed by hand with ARGS=--resume, not by this).
# It refuses to start a run while the box has less than MIN_AVAIL_GIB available: a
# PersonaPlex train peaks at ~50 GiB (60 eager) of the GB10's 121 GiB unified memory, and
# the box has hung twice when other GPU tenants left less than that free.
set -eu
IMAGE="${IMAGE:?set IMAGE to the tag smoke-train pushed (sparks status <job> names it)}"
HOST="${SPARKS_HOST:?set SPARKS_HOST (vlad@192.168.1.140 on the LAN)}"
MIN_AVAIL_GIB="${MIN_AVAIL_GIB:-70}"
FROM="${FROM:-}"

SLEEP="${LADDER_SLEEP:-60}"

# Refuse to submit while the box has less than MIN_AVAIL_GIB available (used before every submit).
gate() {  # gate <run name>
  gib=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "awk '/MemAvailable/ {print int(\$2 / 1048576)}' /proc/meminfo") || gib=""
  case "$gib" in
    ''|*[!0-9]*) echo "$1: could not read MemAvailable from $HOST over ssh"; exit 1 ;;
  esac
  [ "$gib" -ge "$MIN_AVAIL_GIB" ] || { echo "$1: box has $gib GiB available, need $MIN_AVAIL_GIB"; exit 1; }
}

job_state() {  # the job's state name from `sparks status --json`, empty when unreadable
  sparks status --json "$1" 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["state"]["state"])' 2>/dev/null || true
}

# sparks wait exits 0 finished, 75 its --timeout, 1 failed/cancelled/aborted OR an ssh/client
# error. Only a terminal state from the job's own record ends the ladder; a blip does not.
wait_job() {
  unreadable=0
  while :; do
    rc=0; sparks wait --timeout 600 "$1" || rc=$?
    case "$rc" in 0) return 0 ;; 75) continue ;; esac
    st=$(job_state "$1")
    case "$st" in
      queued|building|running) unreadable=0 ;;
      finished) return 0 ;;
      failed|cancelled|aborted) echo "job $1 ended $st"; return 1 ;;
      *)
        unreadable=$((unreadable + 1))
        if [ "$unreadable" -gt 15 ]; then echo "job $1: state unreadable 15 times in a row (sparks wait rc=$rc); giving up, it may still be running"; return 1; fi
        sleep "$SLEEP" ;;
    esac
  done
}

if [ -n "$FROM" ]; then
  found=""
  for b in personaplex moshiko; do for sl in 8 4 2 1; do [ "ladder-$b-$sl" = "$FROM" ] && found=1; done; done
  [ -n "$found" ] || { echo "FROM=$FROM names no run (ladder-{personaplex,moshiko}-{8,4,2,1})"; exit 2; }
fi

for base in personaplex moshiko; do
  for slice in 8 4 2 1; do
    name="ladder-$base-$slice"
    if [ -n "$FROM" ]; then [ "$name" = "$FROM" ] && FROM="" || continue; fi
    gate "$name"
    JOB=$(IMAGE="$IMAGE" make -s train NAME=$name BASE=$base SLICE=$slice RENDER=render-1 EPOCHS=8)
    wait_job "$JOB" || exit 1
    gate "$name (measure)"
    MJOB=$(IMAGE="$IMAGE" make -s measure NAME=$name BASE=$base RUN=$name RENDER=render-1)
    wait_job "$MJOB" || exit 1
  done
done
make bar BOX="$HOST"
