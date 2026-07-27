"""Pluggable coding-worker dispatch: run_coding_task() hands a task off to
whichever CLI coding agent the run was configured with. Both backends
implement the same (task, workdir, timeout) -> final-report-string contract,
so callers (the Research Agent's run_experiment_task tool) don't need to know
which one is actually doing the work.
"""

from typing import Optional

from ai_scientist.tools.claude_code_worker import run_claude_code_task
from ai_scientist.tools.codex_worker import run_codex_task

WORKERS = ("codex", "claude-code")


def run_coding_task(
    task: str,
    workdir: str,
    worker: str = "codex",
    timeout: int = 3600,
    codex_profile: Optional[str] = "fugu",
) -> str:
    if worker == "codex":
        return run_codex_task(task, workdir, timeout=timeout, profile=codex_profile)
    if worker == "claude-code":
        return run_claude_code_task(task, workdir, timeout=timeout)
    raise ValueError(f"Unknown coding worker {worker!r}; choose one of {WORKERS}")
