#!/bin/sh
# One queue, selected by LADDER (default 1). LADDER=1: eight runs, both bases over the four
# rungs of render-1 (ladder-{personaplex,moshiko}-{8,4,2,1}, EPOCHS=8). LADDER=2: the recipe
# sweep, five personaplex SLICE=1 EPOCHS=3 runs, each with its own ARGS (ladder2-pp-s1
# --seed 1, -lr5e-6 --lr 5e-6, -lr1e-4 --lr 1e-4, -r64 --rank 64 --scaling 0.5, -r4
# --rank 4 --scaling 8.0). Each train is followed by its measure (no ARGS: measure reads the
# setup from the checkpoint). An unknown LADDER, or a FROM outside the selected list, exits 2.
# Runs are submitted one at a time, so a box hang loses at most one; FROM=<run name> skips
# the runs before it in the selected ladder (a hung train is resumed by hand with ARGS=--resume, not by this).
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
LADDER="${LADDER:-1}"

SLEEP="${LADDER_SLEEP:-60}"

say() { echo "$(date '+%m-%d %H:%M') $*"; }

# Block until the box has MIN_AVAIL_GIB available (used before every submit); never exits.
gate() {  # gate <run name>
  while :; do
    gib=$(ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "$HOST" "awk '/MemAvailable/ {print int(\$2 / 1048576)}' /proc/meminfo") || gib=""
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
hit = [j for j in jobs if isinstance(j, dict) and j.get("name") == sys.argv[1]
       and isinstance(j.get("submitted_unix"), (int, float)) and j.get("job_id")
       and j["submitted_unix"] >= int(sys.argv[2])]
if hit:
    print(max(hit, key=lambda j: j["submitted_unix"])["job_id"])
' "$1" "$2" 2>/dev/null
}

# Submit a job through make and set JOB to its id. A failed make may still have queued the job
# (sparks can time out copying --data after accepting it), so look it up before resubmitting.
submit() {  # submit <job name> <make target and args...>
  jname=$1; shift
  t0=$(date +%s)   # pinned: a job that landed on attempt 1 must stay findable on later lookups
  hopeless=0       # consecutive attempts where make failed and a good lookup found nothing
  while :; do
    rc=0; out=$(IMAGE="$IMAGE" make -s "$@") || rc=$?
    # the first token of the last line whose first token is strictly job-<id>
    JOB=$(printf '%s\n' "$out" | awk '$1 ~ /^job-[A-Za-z0-9_-]+$/ {id = $1} END {if (id != "") print id}') || JOB=""
    if [ "$rc" -eq 0 ] && [ -n "$JOB" ]; then say "submitted $jname $JOB"; return 0; fi
    say "$jname: make exited $rc without a usable job id; looking in the queue"
    # a job sparks accepted can show in the queue a little late; give it time before looking
    [ "$rc" -eq 0 ] || sleep "$SLEEP"
    while :; do
      if found=$(queued_job "$jname" $((t0 - 300))); then break; fi
      say "$jname: sparks queue unreadable; retrying in ${SLEEP}s"
      sleep "$SLEEP"
    done
    if [ -n "$found" ]; then JOB=$found; say "adopted $jname $JOB from the queue"; return 0; fi
    if [ "$rc" -ne 0 ]; then
      hopeless=$((hopeless + 1))
      if [ "$hopeless" -ge 20 ]; then
        say "$jname: make failed $hopeless times in a row and the queue never had it; giving up (is vendor/ or data/ missing?)"
        exit 1
      fi
    fi
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
      queued|building|running) sleep "$SLEEP" ;;
      finished) return 0 ;;
      failed|cancelled|aborted) say "job $1 ended $st"; return 1 ;;
      *) say "job $1: state unreadable (sparks wait rc=$rc); retrying in ${SLEEP}s"
         sleep "$SLEEP" ;;
    esac
  done
}

# The run list: one word per run, "name,base,slice,epochs[,arg word...]" (the arg words are
# joined by single spaces into the one ARGS=... make variable). Words, never a `while read`
# loop on stdin: ssh in gate() would read the rest of the list.
case "$LADDER" in
  1) RUNS=""
     for base in personaplex moshiko; do
       for slice in 8 4 2 1; do RUNS="$RUNS ladder-$base-$slice,$base,$slice,8"; done
     done ;;
  2) RUNS="ladder2-pp-s1,personaplex,1,3,--seed,1
ladder2-pp-lr5e-6,personaplex,1,3,--lr,5e-6
ladder2-pp-lr1e-4,personaplex,1,3,--lr,1e-4
ladder2-pp-r64,personaplex,1,3,--rank,64,--scaling,0.5
ladder2-pp-r4,personaplex,1,3,--rank,4,--scaling,8.0" ;;
  *) say "LADDER=$LADDER is not a ladder (1 or 2)"; exit 2 ;;
esac

if [ -n "$FROM" ]; then
  names=""; found=""
  for spec in $RUNS; do
    n=${spec%%,*}; names="$names $n"
    [ "$n" = "$FROM" ] && found=1
  done
  [ -n "$found" ] || { say "FROM=$FROM names no run of LADDER=$LADDER ($names )"; exit 2; }
fi

run_one() {  # run_one <name,base,slice,epochs[,arg word...]>
  oldifs=$IFS; IFS=,
  set -- $1
  IFS=$oldifs
  name=$1; base=$2; slice=$3; epochs=$4; shift 4
  extra=""
  if [ "$#" -gt 0 ]; then extra="ARGS=$*"; fi   # "$*" joins with a space (IFS restored)
  gate "$name"
  # ladder 1 passes no ARGS word at all; ladder 2 passes exactly one (quoted: one argv word)
  if [ -n "$extra" ]; then
    submit "$name" train NAME="$name" BASE="$base" SLICE="$slice" RENDER=render-1 EPOCHS="$epochs" "$extra"
  else
    submit "$name" train NAME="$name" BASE="$base" SLICE="$slice" RENDER=render-1 EPOCHS="$epochs"
  fi
  wait_job "$JOB" || exit 1
  gate "$name (measure)"
  submit "measure-$name" measure NAME="$name" BASE="$base" RUN="$name" RENDER=render-1
  MJOB=$JOB
  wait_job "$MJOB" || exit 1
}

for spec in $RUNS; do
  name=${spec%%,*}
  if [ -n "$FROM" ]; then [ "$name" = "$FROM" ] && FROM="" || continue; fi
  run_one "$spec"
done
n=0
until make bar BOX="$HOST"; do
  n=$((n + 1))
  [ "$n" -lt 10 ] || { say "make bar failed $n times; giving up"; exit 1; }
  say "make bar failed ($n/10); retrying in ${SLEEP}s"
  sleep "$SLEEP"
done
