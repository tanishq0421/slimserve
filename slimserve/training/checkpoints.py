"""Checkpoint discovery for resumable training (Phase 2 robustness).

Long runs on free/preemptible compute get killed mid-flight (session timeout,
OOM, a disconnect). With periodic checkpointing on, re-running the *same* command
should pick up where it left off instead of starting over. This module owns the
one piece of logic that needs testing without a GPU: finding the latest
checkpoint HuggingFace wrote. Restoring it is delegated to the HF Trainer.
"""
from __future__ import annotations

import re
from pathlib import Path

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def latest_checkpoint(output_dir: str) -> str | None:
    """Path of the highest-step ``checkpoint-N`` under ``output_dir``, or None.

    Returns None when the directory doesn't exist or holds no checkpoints — the
    "fresh run" signal the trainer uses to decide whether to resume.
    """
    root = Path(output_dir)
    if not root.is_dir():
        return None
    checkpoints = [
        (int(m.group(1)), str(child))
        for child in root.iterdir()
        if child.is_dir() and (m := _CHECKPOINT_RE.match(child.name))
    ]
    return max(checkpoints)[1] if checkpoints else None
