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

avail_gib() {
  ssh -o ConnectTimeout=15 "$HOST" "awk '/MemAvailable/ {print int(\$2 / 1048576)}' /proc/meminfo"
}

wait_job() {  # sparks wait in 10-minute slices: exit 75 is its timeout, anything else ends
  while :; do
    rc=0; sparks wait --timeout 600 "$1" || rc=$?
    [ "$rc" -eq 75 ] || return "$rc"
  done
}

for base in personaplex moshiko; do
  for slice in 8 4 2 1; do
    name="ladder-$base-$slice"
    if [ -n "$FROM" ]; then [ "$name" = "$FROM" ] && FROM="" || continue; fi
    gib=$(avail_gib)
    [ "$gib" -ge "$MIN_AVAIL_GIB" ] || { echo "$name: box has $gib GiB available, need $MIN_AVAIL_GIB"; exit 1; }
    JOB=$(IMAGE=$IMAGE make -s train NAME=$name BASE=$base SLICE=$slice RENDER=render-1 EPOCHS=8)
    wait_job "$JOB"
    MJOB=$(IMAGE=$IMAGE make -s measure NAME=$name BASE=$base RUN=$name RENDER=render-1)
    wait_job "$MJOB"
  done
done
make bar BOX="$HOST"
