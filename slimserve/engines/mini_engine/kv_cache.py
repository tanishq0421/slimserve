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
from slimserve.engines.mini_engine.block_manager import BlockManager


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
    """vLLM's core idea, simplified: KV lives in a shared pool of fixed-size blocks,
    and each sequence holds a *block table* mapping its logical positions to physical
    blocks. Blocks are handed out on demand by the BlockManager, so a sequence only
    occupies (length / block_size) blocks — no reserving to ``max_len`` up front.

    The physical pool is ``[num_blocks, block_size, n_layers, n_kv_heads, head_dim]``.
    Allocation bookkeeping is delegated to the (unit-tested) BlockManager; this class
    just reads/writes tensors at the physical slots it names.
    """

    def __init__(self, block_size: int, num_blocks: int, n_layers: int,
                 n_kv_heads: int, head_dim: int, device: str = "cpu",
                 dtype=None) -> None:
        import torch

        self.block_size = block_size
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self._bm = BlockManager(block_size, num_blocks)
        shape = (num_blocks, block_size, n_layers, n_kv_heads, head_dim)
        self._k = torch.zeros(shape, device=device, dtype=dtype or torch.float32)
        self._v = torch.zeros(shape, device=device, dtype=dtype or torch.float32)
        self._len: dict[int, int] = {}

    def allocate(self, seq_id: int, num_tokens: int) -> None:
        self._bm.allocate(seq_id, num_tokens)
        self._len[seq_id] = 0

    def append(self, seq_id: int, layer: int, key, value):
        """key/value: ``[n_new, n_kv_heads, head_dim]`` for one layer. Writes each
        new token into its physical slot, then gathers and returns the full cached
        K/V for this sequence+layer (in logical order)."""
        start = self._len[seq_id]
        n = key.shape[0]
        for i in range(n):                       # one physical slot per new token
            block, offset = self._bm.append_slot(seq_id, start + i)
            self._k[block, offset, layer] = key[i]
            self._v[block, offset, layer] = value[i]
        end = start + n
        if layer == self.n_layers - 1:           # advance length once per token-step
            self._len[seq_id] = end

        # Gather the seq's blocks (logical order) into a contiguous [end, n_kv, d]
        # view. A real engine fuses this into the attention kernel; here it's an
        # explicit gather for readability.
        blocks = self._bm.block_table(seq_id)
        k_full = self._k[blocks][:, :, layer].reshape(-1, self.n_kv_heads, self.head_dim)
        v_full = self._v[blocks][:, :, layer].reshape(-1, self.n_kv_heads, self.head_dim)
        return k_full[:end], v_full[:end]

    def free(self, seq_id: int) -> None:
        self._bm.free(seq_id)
        self._len.pop(seq_id, None)

    def utilization(self) -> float:
        """Live tokens / (allocated blocks * block_size) — the fragmentation win."""
        return self._bm.utilization(self._len)
