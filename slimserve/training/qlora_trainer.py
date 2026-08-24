"""QLoRA fine-tuning (Phase 3, Week 6).

Plain supervised fine-tuning of a 4-bit base model with LoRA adapters. This is
both a strong baseline on its own and the fallback if distillation underperforms.
"""
from __future__ import annotations

from slimserve.core.registry import register
from slimserve.training.base import BaseTrainer


@register("trainer", "qlora")
class QLoRATrainer(BaseTrainer):
    def compute_loss(self, batch):
        # standard cross-entropy on the target tool-call tokens
        raise NotImplementedError("Phase 3 Wk6: CE loss on tool-call targets.")
