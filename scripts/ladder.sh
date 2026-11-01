#!/bin/sh
# Eight runs, one queue: both bases over the four rungs of render-1, then one measure each.
# Runs are submitted one at a time, so a box hang loses at most one; FROM=<run name> skips
# the runs before it (a hung train is resumed by hand with ARGS=--resume, not by this).
# It refuses to start a run while the box has less than MIN_AVAIL_GIB available: a
# PersonaPlex train peaks at ~50 GiB (60 eager) of the GB10's 121 GiB unified memory, and
# the box has hung twice when other GPU tenants left less than that free.
# It runs unattended for days and never exits on an ssh/network failure: every line is
# stamped; the gate (ssh down, MemAvailable unreadable, too little memory) retries every
# LADDER_SLEEP seconds forever; a submit that fails or prints no job id first looks the job
# up by name in `sparks queue --all --json` (a submit can land and still report a timeout)
# and adopts it, else resubmits, retrying forever and never resubmitting without a lookup that
# succeeded; waiting retries an unreadable state forever. Only a failed/cancelled/aborted job
# record (exit 1) or `make bar` failing 10 times (exit 1) ends it early.
set -eu
IMAGE="${IMAGE:?set IMAGE to the tag smoke-train pushed (sparks status <job> names it)}"
HOST="${SPARKS_HOST:?set SPARKS_HOST (vlad@192.168.1.140 on the LAN)}"
MIN_AVAIL_GIB="${MIN_AVAIL_GIB:-70}"
FROM="${FROM:-}"

SLEEP="${LADDER_SLEEP:-60}"

say() { echo "$(date '+%m-%d %H:%M') $*"; }

# Block until the box has MIN_AVAIL_GIB available (used before every submit); never exits.
gate() {  # gate <run name>
  while :; do
    gib=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "awk '/MemAvailable/ {print int(\$2 / 1048576)}' /proc/meminfo") || gib=""
    case "$gib" in
      ''|*[!0-9]*) say "$1: could not read MemAvailable from $HOST over ssh; retrying in ${SLEEP}s" ;;
      *)
        [ "$gib" -lt "$MIN_AVAIL_GIB" ] || return 0
        say "$1: box has $gib GiB available, need $MIN_AVAIL_GIB; waiting" ;;
    esac
    sleep "$SLEEP"
  done
}

job_state() {  # the job's state name from `sparks status --json`, empty when unreadable
  sparks status --json "$1" 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["state"]["state"])' 2>/dev/null || true
}

# The job_id of the newest queue entry named $1 submitted at or after unix time $2: empty when
# there is none, non-zero when `sparks queue` fails or its output is not a JSON list.
queued_job() {  # queued_job <name> <since>
  sparks queue --all --json 2>/dev/null | python3 -c '
import json, sys
jobs = json.load(sys.stdin)
if not isinstance(jobs, list):
    raise SystemExit(1)
hit = [j for j in jobs if j["name"] == sys.argv[1] and j["submitted_unix"] >= int(sys.argv[2])]
if hit:
    print(max(hit, key=lambda j: j["submitted_unix"])["job_id"])
' "$1" "$2" 2>/dev/null
}

# Submit a job through make and set JOB to its id. A failed make may still have queued the job
# (sparks can time out copying --data after accepting it), so look it up before resubmitting.
submit() {  # submit <job name> <make target and args...>
  jname=$1; shift
  while :; do
    t0=$(date +%s)
    rc=0; out=$(IMAGE="$IMAGE" make -s "$@") || rc=$?
    JOB=$(printf '%s\n' "$out" | grep '^job-' | tail -n 1) || JOB=""
    if [ "$rc" -eq 0 ] && [ -n "$JOB" ]; then say "submitted $jname $JOB"; return 0; fi
    say "$jname: make exited $rc without a usable job id; looking in the queue"
    while :; do
      if found=$(queued_job "$jname" $((t0 - 300))); then break; fi
      say "$jname: sparks queue unreadable; retrying in ${SLEEP}s"
      sleep "$SLEEP"
    done
    if [ -n "$found" ]; then JOB=$found; say "adopted $jname $JOB from the queue"; return 0; fi
    say "$jname: not in the queue; resubmitting in ${SLEEP}s"
    sleep "$SLEEP"
  done
}

# sparks wait exits 0 finished, 75 its --timeout, 1 failed/cancelled/aborted OR an ssh/client
# error. Only a terminal state from the job's own record ends the ladder; a blip does not.
wait_job() {
  while :; do
    rc=0; sparks wait --timeout 600 "$1" || rc=$?
    case "$rc" in 0) return 0 ;; 75) continue ;; esac
    st=$(job_state "$1")
    case "$st" in
      queued|building|running) ;;
      finished) return 0 ;;
      failed|cancelled|aborted) say "job $1 ended $st"; return 1 ;;
      *) say "job $1: state unreadable (sparks wait rc=$rc); retrying in ${SLEEP}s"
         sleep "$SLEEP" ;;
    esac
  done
}

if [ -n "$FROM" ]; then
  found=""
  for b in personaplex moshiko; do for sl in 8 4 2 1; do [ "ladder-$b-$sl" = "$FROM" ] && found=1; done; done
  [ -n "$found" ] || { say "FROM=$FROM names no run (ladder-{personaplex,moshiko}-{8,4,2,1})"; exit 2; }
fi

for base in personaplex moshiko; do
  for slice in 8 4 2 1; do
    name="ladder-$base-$slice"
    if [ -n "$FROM" ]; then [ "$name" = "$FROM" ] && FROM="" || continue; fi
    gate "$name"
    submit "$name" train NAME=$name BASE=$base SLICE=$slice RENDER=render-1 EPOCHS=8
    wait_job "$JOB" || exit 1
    gate "$name (measure)"
    submit "measure-$name" measure NAME=$name BASE=$base RUN=$name RENDER=render-1
    wait_job "$JOB" || exit 1
  done
done
n=0
until make bar BOX="$HOST"; do
  n=$((n + 1))
  [ "$n" -lt 10 ] || { say "make bar failed $n times; giving up"; exit 1; }
  say "make bar failed ($n/10); retrying in ${SLEEP}s"
  sleep "$SLEEP"
done
