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
    def __init__(self, max_len: int, num_heads: int, head_dim: int) -> None:
        self.max_len = max_len
        # TODO Phase 2 Wk3: allocate [num_seqs, max_len, num_heads, head_dim].
        raise NotImplementedError

    def allocate(self, seq_id: int, num_tokens: int) -> None: ...
    def append(self, seq_id: int, key, value) -> None: ...
    def free(self, seq_id: int) -> None: ...
    def utilization(self) -> float: ...


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
