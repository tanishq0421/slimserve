"""Sequence-level knowledge distillation (Phase 2, Week 7).

The production-common flavor: the teacher generates tool-call completions, and
the student is trained (plain SFT) on those completions. No teacher logits
needed at train time — you only need the teacher's *outputs*.
"""
from __future__ import annotations

from slimserve.core.config import DistillConfig
from slimserve.core.interfaces import DistillStrategy
from slimserve.core.registry import register


@register("distill", "sequence_kd")
class SequenceLevelKD(DistillStrategy):
    def compute_loss(self, student_batch, teacher_batch, config: DistillConfig):
        # cross-entropy of the student against the teacher's generated tokens
        raise NotImplementedError("Phase 2 Wk7: CE against teacher completions.")
