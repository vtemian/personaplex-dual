import json
import os
import re
import stat
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "ladder.sh"

POP = (
    'pop() { f="$1"; [ -s "$f" ] || return 1; head -n1 "$f";'
    ' tail -n +2 "$f" > "$f.t"; mv "$f.t" "$f"; }\n'
)
FAKES = {
    "sparks": POP
    + """echo "sparks $*" >> "$FAKE/log"
case "$1" in
  wait) line=$(pop "$FAKE/wait") || line="0"; exit "$line" ;;
  status) line=$(pop "$FAKE/status") || line="1 "
          rc=${line%% *}; out=${line#* }; [ -z "$out" ] || echo "$out"; exit "$rc" ;;
  queue) if [ -e "$FAKE/late" ] && ! grep -q '^sleep ' "$FAKE/log"; then
           echo "[]"; exit 0; fi
         line=$(pop "$FAKE/queue") || line="0 []"
         rc=${line%% *}; out=${line#* }; [ -z "$out" ] || echo "$out"; exit "$rc" ;;
esac
""",
    "ssh": POP
    + """echo "ssh $*" >> "$FAKE/log"
[ ! -e "$FAKE/ssh_fail" ] || exit 255
if line=$(pop "$FAKE/ssh"); then
  [ "$line" != fail ] || exit 255
  echo "$line"; exit 0
fi
cat "$FAKE/mem" 2>/dev/null || echo 100
""",
    "make": POP
    + """echo "make IMAGE=$IMAGE $*" >> "$FAKE/log"
{ echo "--"; for a in "$@"; do printf '%s\n' "$a"; done; } >> "$FAKE/argv"
mode=$(pop "$FAKE/make") || mode=ok
if [ "$mode" = fail ]; then
  echo "sparks: could not copy --data: Operation timed out" >&2; exit 1
fi
[ "$mode" != noid ] || exit 0
for a in "$@"; do
  case "$a" in
    train|measure)
      if [ "$mode" = noisy ]; then
        echo "building image..."; echo "job-stale-one earlier"; echo "note: job-nope"
        echo "job-$a queued at 12:00 (position 1)"; echo "job-bad! junk"; echo "done"
      else
        echo "job-$a"
      fi ;;
  esac
done
""",
    "sleep": """echo "sleep $*" >> "$FAKE/log"
""",
    "date": """case "$1" in
  +%s) n=$(cat "$FAKE/dn" 2>/dev/null || echo 0); echo $((n + 1)) > "$FAKE/dn"
       [ -z "${DATE_JUMP:-}" ] ||
         { echo $(( $(/bin/date +%s) + n * DATE_JUMP )); exit 0; } ;;
esac
exec /bin/date "$@"
""",
}


def state(name):
    return "0 " + json.dumps({"job": {}, "state": {"state": name}, "summary": None})


@pytest.fixture
def box(tmp_path):
    fake = tmp_path / "fake"
    fake.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in FAKES.items():
        p = bindir / name
        p.write_text("#!/bin/sh\n" + body)
        p.chmod(p.stat().st_mode | stat.S_IXUSR)

    def run(
        wait=(), status=(), mem=None, ssh_fail=False, ssh=(), make=(), queue=(), **env
    ):
        (fake / "ssh").write_text("".join(f"{x}\n" for x in ssh))
        (fake / "make").write_text("".join(f"{x}\n" for x in make))
        (fake / "queue").write_text("".join(f"{x}\n" for x in queue))
        (fake / "wait").write_text("".join(f"{x}\n" for x in wait))
        (fake / "status").write_text("".join(f"{x}\n" for x in status))
        if mem is not None:
            (fake / "mem").write_text(f"{mem}\n")
        if ssh_fail:
            (fake / "ssh_fail").touch()
        full = {
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "FAKE": str(fake),
            "IMAGE": "img:tag",
            "SPARKS_HOST": "vlad@box",
            "LADDER_SLEEP": "0",
            **env,
        }
        full.pop("FROM", None) if "FROM" not in env else None
        r = subprocess.run(
            ["sh", str(SCRIPT)],
            capture_output=True,
            text=True,
            env=full,
            cwd=ROOT,
            timeout=60,
        )
        log = (fake / "log").read_text().splitlines() if (fake / "log").exists() else []
        return r, log

    return run


def submits(log):
    return [
        line
        for line in log
        if line.startswith("make") and ("train" in line or "measure" in line)
    ]


def test_happy_path_runs_all_eight(box):
    r, log = box()
    assert r.returncode == 0, r.stderr
    assert len(submits(log)) == 16
    assert " bar BOX=vlad@box" in log[-1]


def test_ssh_blip_in_wait_does_not_end_the_ladder(box):
    # first wait: rc 1, status unreadable, then the job is running, then wait finishes
    r, log = box(
        wait=["1", "0", "0"],
        status=["255 ", state("running")],
        FROM="ladder-moshiko-1",
    )
    assert r.returncode == 0, r.stderr
    assert len([x for x in log if x.startswith("sparks wait")]) == 3
    assert len(submits(log)) == 2


def test_failed_job_stops_the_ladder_before_measure(box):
    r, log = box(wait=["1"], status=[state("failed")], FROM="ladder-moshiko-1")
    assert r.returncode != 0
    assert "failed" in r.stdout + r.stderr
    assert len(submits(log)) == 1
    assert "measure" not in " ".join(submits(log))


def test_bad_from_exits_2_and_submits_nothing(box):
    r, log = box(FROM="ladder-nope-3")
    assert r.returncode == 2
    assert "ladder-nope-3" in r.stdout + r.stderr
    assert submits(log) == []


def test_low_memory_gate_also_guards_the_measure_submit(box):
    # memory is fine for the train, then the box fills up before the measure, then frees
    r, log = box(wait=["0"], ssh=["100", "5", "5", "100"], FROM="ladder-moshiko-1")
    assert r.returncode == 0, r.stderr
    assert (
        "ladder-moshiko-1 (measure): box has 5 GiB available, need 70; waiting"
        in r.stdout
    )
    sub = submits(log)
    assert len(sub) == 2 and "train" in sub[0] and "measure" in sub[1]
    assert len([x for x in log if x.startswith("ssh")]) == 4


def test_low_memory_waits_then_submits(box):
    r, log = box(ssh=["10", "20", "100"], FROM="ladder-moshiko-1")
    assert r.returncode == 0, r.stderr
    assert "ladder-moshiko-1: box has 10 GiB available, need 70; waiting" in r.stdout
    assert "box has 20 GiB available" in r.stdout
    assert len(submits(log)) == 2


@pytest.mark.parametrize("bad", ["", "garbage", "fail"])
def test_unreadable_memory_or_ssh_failure_retries(box, bad):
    r, log = box(ssh=[bad, bad, "100"], FROM="ladder-moshiko-1")
    assert r.returncode == 0, r.stderr
    assert "could not read MemAvailable" in r.stdout
    first_make = next(i for i, x in enumerate(log) if x.startswith("make"))
    assert len([x for x in log[:first_make] if x.startswith("ssh")]) == 3
    ssh = next(x for x in log if x.startswith("ssh"))
    assert "BatchMode=yes" in ssh and "ConnectTimeout=15" in ssh
    assert "ServerAliveInterval=15" in ssh and "ServerAliveCountMax=4" in ssh
    assert len(submits(log)) == 2


def test_every_line_is_stamped_and_submit_is_announced(box):
    r, _ = box(ssh=["10", "100"], FROM="ladder-moshiko-1")
    lines = r.stdout.splitlines()
    assert lines and all(re.match(r"\d\d-\d\d \d\d:\d\d \S", x) for x in lines), lines
    assert any(x[12:] == "submitted ladder-moshiko-1 job-train" for x in lines)
    assert any(
        x[12:] == "submitted measure-ladder-moshiko-1 job-measure" for x in lines
    )


def queue_json(name, job_id, age=0, state="queued"):
    return "0 " + json.dumps(
        [
            {
                "job_id": "job-old",
                "name": name,
                "state": "finished",
                "submitted_unix": 1,
            },
            {
                "job_id": job_id,
                "name": name,
                "state": state,
                "submitted_unix": int(time.time()) - age,
            },
        ]
    )


def test_failed_submit_adopts_the_job_already_in_the_queue(box):
    r, log = box(
        make=["fail"],
        queue=[queue_json("ladder-moshiko-1", "job-adopted")],
        FROM="ladder-moshiko-1",
    )
    assert r.returncode == 0, r.stderr
    assert "adopt" in r.stdout and "job-adopted" in r.stdout
    assert len([x for x in submits(log) if "train" in x]) == 1
    assert "sparks wait --timeout 600 job-adopted" in log


def test_a_late_appearing_job_is_adopted_not_resubmitted(box, tmp_path):
    # sparks accepted the job but the queue shows it only after a sleep has happened
    (tmp_path / "fake").joinpath("late").touch()
    r, log = box(
        make=["fail"],
        queue=[queue_json("ladder-moshiko-1", "job-late")],
        FROM="ladder-moshiko-1",
    )
    assert r.returncode == 0, r.stderr
    assert "adopt" in r.stdout and "job-late" in r.stdout
    assert len([x for x in submits(log) if "train" in x]) == 1
    assert "sparks wait --timeout 600 job-late" in log


def test_failed_submit_with_nothing_queued_resubmits(box):
    r, log = box(make=["fail", "noid"], queue=["0 []", "0 []"], FROM="ladder-moshiko-1")
    assert r.returncode == 0, r.stderr
    assert len([x for x in submits(log) if "train" in x]) == 3
    assert "sparks wait --timeout 600 job-train" in log


def test_old_or_other_named_jobs_are_not_adopted(box):
    stale = queue_json("ladder-moshiko-1", "job-stale", age=1000)
    other = queue_json("ladder-moshiko-2", "job-other")
    r, log = box(make=["fail"], queue=[stale, other], FROM="ladder-moshiko-1")
    assert r.returncode == 0, r.stderr
    assert "job-stale" not in " ".join(log) and "job-other" not in " ".join(log)
    assert len([x for x in submits(log) if "train" in x]) == 2


def test_a_measure_job_is_looked_up_by_its_own_name(box):
    r, log = box(
        wait=["0", "0"],
        make=["ok", "fail"],
        queue=[queue_json("measure-ladder-moshiko-1", "job-m")],
        FROM="ladder-moshiko-1",
    )
    assert r.returncode == 0, r.stderr
    assert "sparks wait --timeout 600 job-m" in log
    assert len(submits(log)) == 2


@pytest.mark.parametrize("bad", ["255 ", "0 not json", "0 {}"])
def test_unreadable_queue_never_resubmits(box, bad):
    r, log = box(make=["fail"], queue=[bad, bad, "0 []"], FROM="ladder-moshiko-1")
    assert r.returncode == 0, r.stderr
    assert len([x for x in log if x.startswith("sparks queue --all --json")]) == 3
    # the second train submit comes only after the lookup that succeeded
    trains = [i for i, x in enumerate(log) if x.startswith("make") and "train" in x]
    between = log[trains[0] : trains[1]]
    assert len([x for x in between if x.startswith("sparks queue")]) == 3
    assert len([x for x in submits(log) if "train" in x]) == 2


def test_unreadable_state_keeps_waiting_with_stamped_lines(box):
    r, log = box(
        wait=["1"] * 20 + ["0"],
        status=["255 "] * 20 + [state("finished")],
        FROM="ladder-moshiko-1",
    )
    assert r.returncode == 0, r.stderr
    assert len([x for x in r.stdout.splitlines() if "unreadable" in x]) >= 20
    assert len(submits(log)) == 2


def test_bar_is_retried_ten_times_then_fails(box, tmp_path):
    make = tmp_path / "bin" / "make"
    make.write_text(
        '#!/bin/sh\necho "make $*" >> "$FAKE/log"\n'
        'case "$*" in *bar*) exit 1 ;; *train*) echo job-train ;;'
        " *measure*) echo job-measure ;; esac\n"
    )
    r, log = box(FROM="ladder-moshiko-1")
    assert r.returncode == 1
    assert len([x for x in log if " bar " in x]) == 10


def test_bar_succeeds_after_a_retry(box, tmp_path):
    make = tmp_path / "bin" / "make"
    make.write_text(
        '#!/bin/sh\necho "make $*" >> "$FAKE/log"\n'
        'case "$*" in *bar*) [ -e "$FAKE/b" ] || { touch "$FAKE/b"; exit 1; } ;;'
        " *train*) echo job-train ;; *measure*) echo job-measure ;; esac\n"
    )
    r, log = box(FROM="ladder-moshiko-1")
    assert r.returncode == 0
    assert len([x for x in log if " bar " in x]) == 2


def test_from_skips_earlier_runs(box):
    r, log = box(FROM="ladder-moshiko-2")
    assert r.returncode == 0
    names = [x for x in submits(log) if "train" in x]
    assert len(names) == 2
    assert "NAME=ladder-moshiko-2" in names[0]
    assert not any("personaplex" in x for x in submits(log))


def test_image_is_passed_quoted_to_make(box):
    _, log = box(IMAGE="img with space:tag", FROM="ladder-moshiko-1")
    assert any("IMAGE=img with space:tag" in x for x in submits(log))


def test_the_lookup_window_is_pinned_at_the_first_attempt(box):
    # each date +%s call jumps 1000s: a per-attempt t0 would miss the attempt-1 job
    r, log = box(
        make=["fail", "fail"],
        queue=["0 []", queue_json("ladder-moshiko-1", "job-landed")],
        FROM="ladder-moshiko-1",
        DATE_JUMP="1000",
    )
    assert r.returncode == 0, r.stderr
    assert "adopted ladder-moshiko-1 job-landed" in r.stdout
    assert len([x for x in submits(log) if "train" in x]) == 2


def test_a_hopeless_submit_gives_up_after_twenty_empty_lookups(box):
    r, log = box(make=["fail"] * 30, FROM="ladder-moshiko-1")
    assert r.returncode == 1
    assert len([x for x in submits(log) if "train" in x]) == 20
    assert "ladder-moshiko-1" in r.stdout and "20" in r.stdout


def test_job_id_is_the_first_token_of_the_last_strict_job_line(box):
    r, log = box(make=["noisy", "noisy"], FROM="ladder-moshiko-1")
    assert r.returncode == 0, r.stderr
    assert "submitted ladder-moshiko-1 job-train" in r.stdout
    assert "sparks wait --timeout 600 job-train" in log
    assert "sparks wait --timeout 600 job-measure" in log


def test_wait_sleeps_when_wait_fails_fast_on_a_live_job(box):
    r, log = box(
        wait=["1", "0", "0"],
        status=[state("running")],
        FROM="ladder-moshiko-1",
        LADDER_SLEEP="7",
    )
    assert r.returncode == 0, r.stderr
    waits = [i for i, x in enumerate(log) if x.startswith("sparks wait")]
    assert "sleep 7" in log[waits[0] : waits[1]]


def test_queue_entries_missing_fields_are_skipped(box):
    junk = [
        {"job_id": "job-x", "submitted_unix": int(time.time())},
        {"job_id": "job-y", "name": "ladder-moshiko-1", "submitted_unix": None},
        {"job_id": "job-z", "name": "ladder-moshiko-1"},
    ]
    good = json.loads(queue_json("ladder-moshiko-1", "job-good")[2:])
    r, _log = box(
        make=["fail"], queue=["0 " + json.dumps(junk + good)], FROM="ladder-moshiko-1"
    )
    assert r.returncode == 0, r.stderr
    assert "adopted ladder-moshiko-1 job-good" in r.stdout


LADDER2 = [
    ("ladder2-pp-s1", "--seed 1"),
    ("ladder2-pp-lr5e-6", "--lr 5e-6"),
    ("ladder2-pp-lr1e-4", "--lr 1e-4"),
    ("ladder2-pp-r64", "--rank 64 --scaling 0.5"),
    ("ladder2-pp-r4", "--rank 4 --scaling 8.0"),
]


def expected_ladder2(runs):
    out = []
    for name, args in runs:
        out.append(
            f"make IMAGE=img:tag -s train NAME={name} BASE=personaplex SLICE=1"
            f" RENDER=render-1 EPOCHS=3 ARGS={args}"
        )
        out.append(
            f"make IMAGE=img:tag -s measure NAME={name} BASE=personaplex"
            f" RUN={name} RENDER=render-1"
        )
    return out


def test_ladder_2_submits_five_trains_each_followed_by_its_measure(box):
    r, log = box(LADDER="2")
    assert r.returncode == 0, r.stderr
    assert submits(log) == expected_ladder2(LADDER2)
    assert " bar BOX=vlad@box" in log[-1]


def test_ladder_2_from_starts_there(box):
    r, log = box(LADDER="2", FROM="ladder2-pp-r64")
    assert r.returncode == 0, r.stderr
    assert submits(log) == expected_ladder2(LADDER2[3:])


def test_ladder_2_gates_before_every_submit(box):
    r, log = box(LADDER="2", FROM="ladder2-pp-r4")
    assert r.returncode == 0, r.stderr
    assert len([x for x in log if x.startswith("ssh")]) == 2


def test_ladder_2_rejects_a_ladder_1_run_name(box):
    r, log = box(LADDER="2", FROM="ladder-moshiko-1")
    assert r.returncode == 2
    assert "ladder2-pp-r64" in r.stdout + r.stderr
    assert submits(log) == []


def test_ladder_1_rejects_a_ladder_2_run_name(box):
    r, log = box(FROM="ladder2-pp-s1")
    assert r.returncode == 2
    assert submits(log) == []


def test_unknown_ladder_exits_2(box):
    r, log = box(LADDER="9")
    assert r.returncode == 2
    assert "LADDER" in r.stdout + r.stderr
    assert submits(log) == []


def test_ladder_1_default_commands_are_unchanged(box):
    r, log = box(FROM="ladder-personaplex-1")
    assert r.returncode == 0, r.stderr
    assert submits(log)[0] == (
        "make IMAGE=img:tag -s train NAME=ladder-personaplex-1 BASE=personaplex"
        " SLICE=1 RENDER=render-1 EPOCHS=8"
    )


def argv_calls(tmp_path):
    """The fake make's argv, one list per call, one entry per real argv word."""
    calls, cur = [], None
    for x in (tmp_path / "fake" / "argv").read_text().split("\n")[:-1]:
        if x == "--":
            cur = []
            calls.append(cur)
        else:
            cur.append(x)
    return calls


def test_ladder_2_train_carries_exactly_one_args_word(box, tmp_path):
    r, _ = box(LADDER="2")
    assert r.returncode == 0, r.stderr
    trains = [c for c in argv_calls(tmp_path) if "train" in c]
    assert len(trains) == len(LADDER2)
    for call, (_, args) in zip(trains, LADDER2, strict=True):
        words = [w for w in call if w.startswith("ARGS=")]
        assert words == [f"ARGS={args}"]
    assert "ARGS=--rank 64 --scaling 0.5" in trains[3]


def test_ladder_3_submits_the_pilot_then_its_measure(box):
    r, log = box(LADDER="3")
    assert r.returncode == 0, r.stderr
    assert submits(log) == [
        "make IMAGE=img:tag -s train NAME=cons1-pp-l10 BASE=personaplex SLICE=1"
        " RENDER=render-1 EPOCHS=3 ARGS=--consistency 10",
        "make IMAGE=img:tag -s measure NAME=cons1-pp-l10 BASE=personaplex"
        " RUN=cons1-pp-l10 RENDER=render-1",
    ]
    assert " bar BOX=vlad@box" in log[-1]


def test_ladder_3_train_carries_exactly_one_args_word(box, tmp_path):
    r, _ = box(LADDER="3")
    assert r.returncode == 0, r.stderr
    (train,) = [c for c in argv_calls(tmp_path) if "train" in c]
    assert [w for w in train if w.startswith("ARGS=")] == ["ARGS=--consistency 10"]


def test_ladder_3_rejects_a_ladder_2_run_name(box):
    r, log = box(LADDER="3", FROM="ladder2-pp-s1")
    assert r.returncode == 2
    assert "cons1-pp-l10" in r.stdout + r.stderr
    assert submits(log) == []


def test_ladder_4_submits_the_lambda_3_pilot_then_its_measure(box):
    r, log = box(LADDER="4")
    assert r.returncode == 0, r.stderr
    assert submits(log) == [
        "make IMAGE=img:tag -s train NAME=cons1-pp-l3 BASE=personaplex SLICE=1"
        " RENDER=render-1 EPOCHS=3 ARGS=--consistency 3",
        "make IMAGE=img:tag -s measure NAME=cons1-pp-l3 BASE=personaplex"
        " RUN=cons1-pp-l3 RENDER=render-1",
    ]
    assert " bar BOX=vlad@box" in log[-1]


def test_ladder_4_rejects_a_ladder_3_run_name(box):
    r, log = box(LADDER="4", FROM="cons1-pp-l10")
    assert r.returncode == 2
    assert "cons1-pp-l3" in r.stdout + r.stderr
    assert submits(log) == []


def test_ladder_6_runs_the_voiced_teacher_on_render_2(box):
    r, log = box(LADDER="6")
    assert r.returncode == 0, r.stderr
    assert submits(log) == [
        "make IMAGE=img:tag -s train NAME=voiced1-pp-l3 BASE=personaplex SLICE=1"
        " RENDER=render-2 EPOCHS=3 ARGS=--consistency 3 --teacher voiced",
        "make IMAGE=img:tag -s measure NAME=voiced1-pp-l3 BASE=personaplex"
        " RUN=voiced1-pp-l3 RENDER=render-2",
    ]
    assert " bar BOX=vlad@box" in log[-1]


def test_ladder_6_train_carries_exactly_one_args_word(box, tmp_path):
    r, _ = box(LADDER="6")
    assert r.returncode == 0, r.stderr
    (train,) = [c for c in argv_calls(tmp_path) if "train" in c]
    assert [w for w in train if w.startswith("ARGS=")] == [
        "ARGS=--consistency 3 --teacher voiced"
    ]


def test_ladder_6_rejects_a_ladder_4_run_name(box):
    r, log = box(LADDER="6", FROM="cons1-pp-l3")
    assert r.returncode == 2
    assert "voiced1-pp-l3" in r.stdout + r.stderr
    assert submits(log) == []
