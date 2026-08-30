"""CPU unit tests for the logit-KD loss — no GPU, no model, tiny tensors.

These pin the three things that are easy to get subtly wrong: the CE reduces to
plain cross-entropy at alpha=0, a student that already matches the teacher has
zero KD, and completion-only masking means prompt positions never touch the loss.
"""
import torch
import torch.nn.functional as F

from slimserve.core.config import DistillConfig
from slimserve.training.distillation.logit_kd import LogitKD

V, K = 6, 3


def _cfg(temperature=2.0, alpha=0.5):
    return DistillConfig(strategy="logit_kd", teacher_model="x",
                         temperature=temperature, alpha=alpha)


def _batch():
    """B=1, L=4, V=6. Prompt token at pos 0 is masked; positions 1-3 supervised."""
    torch.manual_seed(0)
    logits = torch.randn(1, 4, V)
    labels = torch.tensor([[-100, 2, 3, 4]])
    # teacher top-k over the 3 supervised positions (valid vocab indices < V)
    topk_ids = torch.tensor([[[0, 1, 2], [1, 2, 3], [2, 3, 4]]])   # [1, 3, K]
    topk_vals = torch.tensor([[[3.0, 1.0, 0.0],
                               [2.0, 2.0, 0.0],
                               [1.0, 0.0, -1.0]]])                  # [1, 3, K]
    student = {"logits": logits, "labels": labels}
    teacher = {"topk_ids": topk_ids, "topk_vals": topk_vals}
    return student, teacher


def test_alpha_zero_is_plain_cross_entropy():
    student, teacher = _batch()
    loss = LogitKD().compute_loss(student, teacher, _cfg(alpha=0.0))
    shift_logits = student["logits"][:, :-1, :]
    shift_labels = student["labels"][:, 1:]
    ce = F.cross_entropy(shift_logits.reshape(-1, V), shift_labels.reshape(-1),
                         ignore_index=-100)
    assert torch.allclose(loss, ce, atol=1e-6)


def test_matched_student_has_zero_kd():
    student, teacher = _batch()
    # Force the student's logits at the teacher's top-k indices to equal the
    # teacher values, for every supervised position -> identical distributions.
    logits = student["logits"].clone()
    for i, pos in enumerate((0, 1, 2)):          # shift rows == logits rows 0,1,2
        logits[0, pos, teacher["topk_ids"][0, i]] = teacher["topk_vals"][0, i]
    student["logits"] = logits
    # alpha=1 -> loss is purely the KD term, which should vanish.
    loss = LogitKD().compute_loss(student, teacher, _cfg(alpha=1.0))
    assert torch.allclose(loss, torch.zeros(()), atol=1e-6)


def test_alpha_blends_linearly():
    student, teacher = _batch()
    a0 = LogitKD().compute_loss(student, teacher, _cfg(alpha=0.0))
    a1 = LogitKD().compute_loss(student, teacher, _cfg(alpha=1.0))
    half = LogitKD().compute_loss(student, teacher, _cfg(alpha=0.5))
    assert torch.allclose(half, 0.5 * (a0 + a1), atol=1e-6)


def test_masked_positions_do_not_affect_loss():
    # Middle position is masked in the labels; perturbing its logits must not move
    # the loss (ignored by CE, never selected by KD).
    torch.manual_seed(1)
    logits = torch.randn(1, 4, V)
    labels = torch.tensor([[-100, 2, -100, 4]])   # shift_labels = [2, -100, 4]
    topk_ids = torch.tensor([[[0, 1, 2], [3, 4, 5]]])            # 2 supervised rows
    topk_vals = torch.tensor([[[2.0, 1.0, 0.0], [1.0, 0.5, 0.0]]])
    teacher = {"topk_ids": topk_ids, "topk_vals": topk_vals}
    before = LogitKD().compute_loss({"logits": logits.clone(), "labels": labels},
                                    teacher, _cfg())
    poked = logits.clone()
    poked[0, 1] += 99.0                            # row 1 is the masked position
    after = LogitKD().compute_loss({"logits": poked, "labels": labels},
                                   teacher, _cfg())
    assert torch.allclose(before, after, atol=1e-6)


def test_mismatched_student_has_positive_kd():
    student, teacher = _batch()
    loss = LogitKD().compute_loss(student, teacher, _cfg(alpha=1.0))
    assert loss.item() > 0.0
