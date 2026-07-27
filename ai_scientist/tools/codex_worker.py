"""Delegate the actual "write code, run it, debug it" work to Codex CLI (a proven
coding agent) instead of hand-rolling a code-gen/execution/debugging loop. The
Research Agent decides *what* to do next; Codex actually does it.

Runs with --dangerously-bypass-approvals-and-sandbox: Codex's own bubblewrap
sandboxing doesn't work in this cluster environment (nested user/network
namespaces aren't permitted), so it relies on the SLURM allocation itself as the
isolation boundary instead. `profile` selects the Codex CLI profile (e.g. the
internal "fugu" profile routed through the Sakana gateway, or None/"" to use
Codex's own default profile/login against the regular OpenAI API).
"""

import base64
import os
import os.path as osp
import subprocess
import time
from typing import Optional


def run_codex_task(task: str, workdir: str, timeout: int = 3600, profile: Optional[str] = "fugu") -> str:
    """Give Codex a concrete coding/experiment task to carry out in workdir
    (write code, run it, fix errors, report results). Returns Codex's final
    report message. Codex can read/write any file under workdir and run shell
    commands there; give it a self-contained, specific task description."""
    os.makedirs(workdir, exist_ok=True)
    last_message_path = osp.join(workdir, f"_codex_last_message_{int(time.time() * 1000)}.txt")
    cmd = ["codex", "exec"]
    if profile:
        cmd += ["--profile", profile]
    cmd += [
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "-C", workdir,
        "-o", last_message_path,
        task,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"Codex task timed out after {timeout} seconds without finishing."

    final_message = "(no final message captured)"
    if osp.exists(last_message_path):
        with open(last_message_path) as f:
            final_message = f.read()
        os.remove(last_message_path)

    if result.returncode != 0:
        return (
            f"Codex exited with code {result.returncode}.\n"
            f"Stderr tail: {result.stderr[-2000:]}\n"
            f"Final message: {final_message}"
        )
    return final_message


def list_workdir_files(workdir: str) -> str:
    """List all files under workdir (recursively), so you know what Codex has
    produced so far (result files, plots, logs, etc.) before deciding what to
    read or what to ask Codex to do next."""
    if not osp.isdir(workdir):
        return f"{workdir} does not exist."
    lines = []
    for root, _dirs, files in os.walk(workdir):
        for fname in files:
            if fname.startswith("_codex_last_message"):
                continue
            full = osp.join(root, fname)
            rel = osp.relpath(full, workdir)
            size = osp.getsize(full)
            lines.append(f"{rel} ({size} bytes)")
    return "\n".join(sorted(lines)) if lines else "(empty)"


def read_text_file(path: str, max_chars: int = 8000) -> str:
    """Read a text file (results.json, logs, etc.), truncated to max_chars."""
    if not osp.isfile(path):
        return f"{path} does not exist."
    try:
        with open(path, "r", errors="replace") as f:
            text = f.read()
    except Exception as e:
        return f"Failed to read {path}: {e}"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated {len(text) - max_chars} characters]"
    return text


def describe_plot(image_path: str, question: str, model: str = "fugu") -> str:
    """Ask a question about a plot/figure image (e.g. 'does the training loss
    converge?', 'is there anything unusual in this figure?')."""
    if not osp.isfile(image_path):
        return f"{image_path} does not exist."
    import openai

    client = openai.OpenAI(
        api_key=os.environ["SAKANA_API_KEY"], base_url="https://api.sakana.ai/v1"
    )
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content
