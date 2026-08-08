"""Run a CLI subprocess in its own process group so a timeout kills the whole
process tree. subprocess.run(timeout=...) only kills the direct child: a coding
agent's own children (e.g. a training script it launched) would survive the
timeout and keep running, competing for GPU/CPU with whatever the pipeline does
next. Here the child gets its own session/process group, and on timeout the
entire group is terminated (SIGTERM, a short grace period, then SIGKILL).
"""

import os
import signal
import subprocess
from typing import List, Optional


def run_in_process_group(
    cmd: List[str], timeout: int, cwd: Optional[str] = None
) -> tuple[int, str, str]:
    """Like subprocess.run(capture_output=True, text=True, timeout=...), returning
    (returncode, stdout, stderr) -- but the child runs as its own process-group
    leader, and a timeout kills the whole group before re-raising
    subprocess.TimeoutExpired."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        raise
    return proc.returncode, stdout, stderr


def _kill_process_group(proc: subprocess.Popen) -> None:
    # start_new_session=True made the child a session/group leader, so its pid
    # is the pgid of everything it (transitively) spawned.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # SIGKILL the group unconditionally: even if the direct child exited on
    # SIGTERM, a grandchild that ignored it would otherwise live on.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.communicate()  # reap the child and drain/close its pipes
