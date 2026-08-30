"""From-scratch KV cache implementations (Phase 2).

Two concrete strategies behind the KVCache interface:

  ContiguousKVCache — the naive approach: one big pre-allocated tensor per
      sequence sized to max_len. Simple, but wastes memory on short sequences
      (internal fragmentation) and can't share.

  PagedKVCache — vLLM's core idea, simplified: KV is stored in fixed-size
      blocks; a per-sequence block table maps logical positions to physical
      blocks. Near-zero fragmentation; blocks are allocated on demand.

Keep this code READABLE over clever — its job is to teach, so light on
abstraction and heavy on comments.
"""
from __future__ import annotations

from slimserve.core.interfaces import KVCache


class ContiguousKVCache(KVCache):
    """Naive: one pre-allocated ``[n_layers, max_len, n_kv_heads, head_dim]`` K and
    V tensor per sequence. Simple, but every sequence reserves ``max_len`` up front
    regardless of how long it actually gets — the internal fragmentation the paged
    cache removes.
    """

    def __init__(self, n_layers: int, max_len: int, n_kv_heads: int,
                 head_dim: int, device: str = "cpu", dtype=None) -> None:
        import torch

        self.n_layers = n_layers
        self.max_len = max_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype or torch.float32
        self._k: dict[int, "torch.Tensor"] = {}
        self._v: dict[int, "torch.Tensor"] = {}
        self._len: dict[int, int] = {}          # live tokens per sequence

    def allocate(self, seq_id: int, num_tokens: int) -> None:
        import torch

        shape = (self.n_layers, self.max_len, self.n_kv_heads, self.head_dim)
        self._k[seq_id] = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self._v[seq_id] = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self._len[seq_id] = 0

    def append(self, seq_id: int, layer: int, key, value):
        """key/value: ``[n_new, n_kv_heads, head_dim]`` for one layer. Writes them at
        the sequence's current length and returns the full cached K/V for the layer.

        The write position is the *current* length for every layer in a step; the
        length only advances after the last layer, so all layers stay aligned.
        """
        start = self._len[seq_id]
        n = key.shape[0]
        self._k[seq_id][layer, start:start + n] = key
        self._v[seq_id][layer, start:start + n] = value
        end = start + n
        if layer == self.n_layers - 1:          # one token-step spans all layers
            self._len[seq_id] = end
        return self._k[seq_id][layer, :end], self._v[seq_id][layer, :end]

    def free(self, seq_id: int) -> None:
        for store in (self._k, self._v, self._len):
            store.pop(seq_id, None)

    def utilization(self) -> float:
        if not self._len:
            return 0.0
        return sum(self._len.values()) / (len(self._len) * self.max_len)


class PagedKVCache(KVCache):
    def __init__(self, block_size: int, num_blocks: int,
                 num_heads: int, head_dim: int) -> None:
        self.block_size = block_size
        self._free_blocks: list[int] = list(range(num_blocks))
        self._block_table: dict[int, list[int]] = {}   # seq_id -> [block ids]
        # TODO Phase 2 Wk4: physical KV pool [num_blocks, block_size, ...].
        raise NotImplementedError

    def allocate(self, seq_id: int, num_tokens: int) -> None: ...
    def append(self, seq_id: int, key, value) -> None: ...
    def free(self, seq_id: int) -> None: ...
    def utilization(self) -> float:
        """Live tokens / (allocated blocks * block_size) — the fragmentation win."""
        ...
