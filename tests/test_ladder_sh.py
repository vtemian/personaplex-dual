import json
import os
import stat
import subprocess
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
esac
""",
    "ssh": """echo "ssh $*" >> "$FAKE/log"
[ ! -e "$FAKE/ssh_fail" ] || exit 255
cat "$FAKE/mem" 2>/dev/null || echo 100
""",
    "make": """echo "make IMAGE=$IMAGE $*" >> "$FAKE/log"
for a in "$@"; do
  case "$a" in
    train) echo job-train ;;
    measure) echo job-measure ;;
  esac
done
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

    def run(wait=(), status=(), mem=None, ssh_fail=False, **env):
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


def test_unreadable_state_gives_up_naming_the_job(box):
    r, log = box(wait=["1"] * 40, status=["255 "] * 40, FROM="ladder-moshiko-1")
    assert r.returncode == 1
    assert "job-train" in r.stdout + r.stderr
    assert len([x for x in log if x.startswith("sparks status")]) <= 16
    assert len(submits(log)) == 1


def test_bad_from_exits_2_and_submits_nothing(box):
    r, log = box(FROM="ladder-nope-3")
    assert r.returncode == 2
    assert "ladder-nope-3" in r.stdout + r.stderr
    assert submits(log) == []


def test_low_memory_exits_before_any_submit(box):
    r, log = box(mem=10)
    assert r.returncode == 1
    assert "10 GiB" in r.stdout + r.stderr
    assert submits(log) == []


def test_low_memory_gate_also_guards_the_measure_submit(box, tmp_path):
    # memory is fine for the train, then the box fills up before the measure
    ssh = tmp_path / "bin" / "ssh"
    ssh.write_text(
        '#!/bin/sh\necho "ssh $*" >> "$FAKE/log"\n'
        'if [ -e "$FAKE/seen" ]; then echo 5; else touch "$FAKE/seen"; echo 100; fi\n'
    )
    r, log = box(wait=["0"], FROM="ladder-moshiko-1")
    assert r.returncode == 1
    assert len(submits(log)) == 1
    assert "measure" not in " ".join(submits(log))


@pytest.mark.parametrize("out", ["", "garbage"])
def test_unreadable_memory_exits_1_naming_run_and_host(box, out):
    r, log = box(mem=out)
    assert r.returncode == 1
    assert "ladder-personaplex-8" in r.stdout + r.stderr
    assert "vlad@box" in r.stdout + r.stderr
    assert submits(log) == []


def test_ssh_failure_exits_1_and_ssh_is_batchmode(box):
    r, log = box(ssh_fail=True)
    assert r.returncode == 1
    assert "vlad@box" in r.stdout + r.stderr
    ssh = next(x for x in log if x.startswith("ssh"))
    assert "BatchMode=yes" in ssh and "ConnectTimeout=15" in ssh
    assert submits(log) == []


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
