"""CPU unit tests for the logit-KD data helpers — the alignment gates.

extract_teacher_topk must pick the right rows (next-token shift + completion-only
mask) and the right top-k; pad_teacher_topk must pad ragged completions without
corrupting the real rows.
"""
import torch

from slimserve.training.dataset import extract_teacher_topk, pad_teacher_topk


def test_extract_picks_topk_at_supervised_positions():
    # rows 0,1,2 each have a clean top-2; all three positions supervised.
    logits = torch.tensor([
        [5.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 9.0, 7.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 8.0, 6.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],   # last row never used (no token after it)
    ])
    labels = [-100, 2, 3, 4]              # shift_labels = [2,3,4] -> all supervised
    vals, ids = extract_teacher_topk(logits, labels, k=2)
    assert ids.shape == (3, 2) and vals.shape == (3, 2)
    assert ids.tolist() == [[0, 1], [2, 3], [4, 5]]
    assert torch.allclose(vals.float(), torch.tensor([[5.0, 4.0], [9.0, 7.0], [8.0, 6.0]]))


def test_extract_skips_masked_positions():
    logits = torch.arange(4 * 6, dtype=torch.float).reshape(4, 6)
    labels = [-100, 2, -100, 4]           # shift_labels = [2,-100,4] -> rows 0 and 2
    vals, ids = extract_teacher_topk(logits, labels, k=3)
    assert ids.shape == (2, 3)            # only two supervised rows
    # row 0 of output is logits row 0; row 1 of output is logits row 2 (row 1 skipped)
    assert ids[0].tolist() == [5, 4, 3]
    assert ids[1].tolist() == [5, 4, 3]  # logits row 2, same descending order


def test_pad_ragged_completions():
    ids = [[[0, 1], [2, 3]], [[1, 2]]]           # ex0 has n=2, ex1 has n=1
    vals = [[[5.0, 4.0], [9.0, 7.0]], [[3.0, 1.0]]]
    topk_ids, topk_vals, kd_mask = pad_teacher_topk(ids, vals)
    assert topk_ids.shape == (2, 2, 2)           # [B=2, M=2, k=2]
    assert kd_mask.tolist() == [[True, True], [True, False]]
    assert topk_ids[1, 0].tolist() == [1, 2]     # ex1 real row preserved
    assert topk_ids[1, 1].tolist() == [0, 0]     # ex1 pad row zeroed
    assert torch.allclose(topk_vals[0], torch.tensor([[5.0, 4.0], [9.0, 7.0]]))


def test_extract_and_pad_roundtrip_aligns_with_loss_mask():
    # The count of supervised rows from extract must equal labels' supervised count,
    # so the loss (which re-derives it) lines up with the stored rows.
    logits = torch.randn(5, 6)
    labels = [-100, -100, 1, 2, 3]        # shift_labels = [-100,1,2,3] -> 3 supervised
    _, ids = extract_teacher_topk(logits, labels, k=2)
    shift_labels = torch.tensor(labels)[1:]
    assert ids.shape[0] == int((shift_labels != -100).sum())
