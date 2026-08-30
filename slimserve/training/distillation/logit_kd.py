"""Logit / soft-label distillation (Phase 3, Week 7 stretch).

Classic KD: minimize KL divergence between the student's and teacher's
temperature-softened output distributions, blended with the task loss via
``alpha``. Requires teacher logits at train time — fine here because we own the
open-weight teacher.

To keep it runnable on a single T4, the teacher's distribution is stored as
**top-k** logits over the *completion* tokens only (see ``build_logits`` /
``extract_teacher_topk``). The student's full-vocab logits are gathered at those
same top-k indices, so both sides form a distribution over the identical support
and the KL is well-defined. Teacher and student share the Qwen2.5 tokenizer, so
token ids line up with no vocab remapping.
"""
from __future__ import annotations

from slimserve.core.config import DistillConfig
from slimserve.core.interfaces import DistillStrategy
from slimserve.core.registry import register


@register("distill", "logit_kd")
class LogitKD(DistillStrategy):
    """L = (1 - alpha) * CE(student, labels) + alpha * T^2 * KL(teacher || student).

    The ``T^2`` factor rescales the KD gradient back to the same magnitude as the
    un-softened loss (Hinton et al.), so ``alpha`` alone controls the blend.
    """

    def compute_loss(self, student_batch, teacher_batch, config: DistillConfig):
        import torch
        import torch.nn.functional as F

        temperature = config.temperature
        alpha = config.alpha
        logits = student_batch["logits"]          # [B, L, V]
        labels = student_batch["labels"]          # [B, L]

        # Next-token shift (HF convention): position i predicts token i+1.
        shift_logits = logits[:, :-1, :]          # [B, L-1, V]
        shift_labels = labels[:, 1:]              # [B, L-1]
        supervised = shift_labels != -100         # completion tokens only

        # --- task loss: cross-entropy on the gold labels ---
        # Reuse the model's own loss when the trainer already computed it.
        ce = student_batch.get("ce_loss")
        if ce is None:
            ce = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )

        # --- KD loss: KL to the teacher's top-k soft labels ---
        # For each example the teacher's top-k rows were stored for exactly the
        # supervised positions, in left-to-right order, so the first ``n`` rows
        # align token-for-token with ``supervised``.
        topk_ids = teacher_batch["topk_ids"]      # [B, M, k] long
        topk_vals = teacher_batch["topk_vals"]    # [B, M, k] float
        kd_terms = []
        for b in range(shift_logits.size(0)):
            n = int(supervised[b].sum())
            if n == 0:
                continue
            student_sup = shift_logits[b][supervised[b]].float()   # [n, V]
            t_ids = topk_ids[b][:n]                                 # [n, k]
            t_vals = topk_vals[b][:n].float()                      # [n, k]
            student_topk = torch.gather(student_sup, 1, t_ids)     # [n, k]
            log_p_student = F.log_softmax(student_topk / temperature, dim=-1)
            p_teacher = F.softmax(t_vals / temperature, dim=-1)    # renormalize over k
            kd_terms.append(
                F.kl_div(log_p_student, p_teacher, reduction="batchmean")
            )
        kd = torch.stack(kd_terms).mean() if kd_terms else logits.new_zeros(())

        return (1 - alpha) * ce + alpha * (temperature ** 2) * kd
