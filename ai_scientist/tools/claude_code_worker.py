"""Delegate the actual "write code, run it, debug it" work to the Claude Code
CLI, as an alternative coding worker to Codex (see codex_worker.py). Runs with
--dangerously-skip-permissions for the same reason codex_worker runs with
--dangerously-bypass-approvals-and-sandbox: the CLI's own sandboxing/approval
flow isn't usable unattended in this cluster environment, so the enclosing
allocation (e.g. a dedicated SLURM allocation or container) is the actual
isolation boundary.

Requires the `claude` CLI on PATH, authenticated (e.g. via `claude auth` or
ANTHROPIC_API_KEY) independently of any other model configured for this run.
"""

import json
import os
import subprocess
from typing import Optional


def run_claude_code_task(task: str, workdir: str, timeout: int = 3600, profile: Optional[str] = None) -> str:
    """Give Claude Code a concrete coding/experiment task to carry out in
    workdir (write code, run it, fix errors, report results). Returns Claude's
    final report message. Same (task, workdir, timeout) contract as
    codex_worker.run_codex_task so the two are interchangeable behind
    coding_worker.run_coding_task. `profile` is accepted for signature parity
    with the Codex worker but currently unused (Claude Code has no CLI-profile
    concept; use its own auth/config for account selection)."""
    os.makedirs(workdir, exist_ok=True)
    cmd = [
        "claude", "-p", task,
        "--dangerously-skip-permissions",
        "--output-format", "json",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=workdir)
    except subprocess.TimeoutExpired:
        return f"Claude Code task timed out after {timeout} seconds without finishing."

    stdout = result.stdout.strip()
    payload = None
    if stdout:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = None

    if payload is not None:
        if payload.get("is_error"):
            return (
                f"Claude Code reported an error: {payload.get('result', '(no message)')}\n"
                f"Stderr tail: {result.stderr[-2000:]}"
            )
        return payload.get("result") or "(no final message captured)"

    if result.returncode != 0:
        return (
            f"Claude Code exited with code {result.returncode}.\n"
            f"Stderr tail: {result.stderr[-2000:]}\n"
            f"Stdout tail: {stdout[-2000:]}"
        )
    return stdout or "(no final message captured)"
