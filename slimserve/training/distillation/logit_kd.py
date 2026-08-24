"""Logit / soft-label distillation (Phase 3, Week 7 stretch).

Classic KD: minimize KL divergence between the student's and teacher's
temperature-softened output distributions, blended with the task loss via
``alpha``. Requires teacher logits at train time — fine here because we own the
open-weight teacher.
"""
from __future__ import annotations

from slimserve.core.config import DistillConfig
from slimserve.core.interfaces import DistillStrategy
from slimserve.core.registry import register


@register("distill", "logit_kd")
class LogitKD(DistillStrategy):
    def compute_loss(self, student_batch, teacher_batch, config: DistillConfig):
        # T = config.temperature
        # kd  = KL(softmax(student/T), softmax(teacher/T)) * T*T
        # task = cross_entropy(student, labels)
        # return config.alpha * kd + (1 - config.alpha) * task
        raise NotImplementedError("Phase 3 Wk7: KL soft-label loss + task loss.")
