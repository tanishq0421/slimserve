"""Block allocator for the paged KV cache (Phase 2).

This is the bookkeeping half of PagedAttention, kept separate from the tensor
storage so it's pure Python and unit-testable with no model or GPU. It hands out
fixed-size physical blocks from a free list and tracks, per sequence, which
physical blocks hold its tokens (the *block table*). Mapping a logical token
position to physical memory is then just:

    logical_block = position // block_size
    offset        = position %  block_size
    physical      = block_table[seq_id][logical_block]

That indirection is what lets sequences grow a block at a time instead of each
reserving a worst-case contiguous slab — the near-zero fragmentation win.
"""
from __future__ import annotations


class BlockManager:
    def __init__(self, block_size: int, num_blocks: int) -> None:
        self.block_size = block_size
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks))
        self._tables: dict[int, list[int]] = {}   # seq_id -> [physical block ids]

    # --- capacity -----------------------------------------------------------
    def num_free_blocks(self) -> int:
        return len(self._free)

    def _blocks_for(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size   # ceil div

    def can_allocate(self, num_tokens: int) -> bool:
        return self._blocks_for(num_tokens) <= len(self._free)

    # --- lifecycle ----------------------------------------------------------
    def allocate(self, seq_id: int, num_tokens: int) -> None:
        """Reserve enough blocks to hold ``num_tokens`` for a new sequence."""
        if seq_id in self._tables:
            raise ValueError(f"seq {seq_id} already allocated")
        need = self._blocks_for(num_tokens)
        if need > len(self._free):
            raise MemoryError(f"need {need} blocks, {len(self._free)} free")
        self._tables[seq_id] = [self._free.pop() for _ in range(need)]

    def append_slot(self, seq_id: int, position: int) -> tuple[int, int]:
        """Physical (block, offset) for logical ``position``; grow the table if the
        position crosses into a not-yet-mapped block (a decode step past the end)."""
        table = self._tables[seq_id]
        logical_block = position // self.block_size
        offset = position % self.block_size
        while logical_block >= len(table):
            if not self._free:
                raise MemoryError("out of KV blocks")
            table.append(self._free.pop())
        return table[logical_block], offset

    def block_table(self, seq_id: int) -> list[int]:
        return list(self._tables[seq_id])

    def free(self, seq_id: int) -> None:
        """Return a finished sequence's blocks to the free list."""
        for block in self._tables.pop(seq_id, []):
            self._free.append(block)

    # --- metric -------------------------------------------------------------
    def utilization(self, live_tokens: dict[int, int]) -> float:
        """Live tokens / allocated slots — the fragmentation measure.

        ``live_tokens`` maps seq_id -> how many tokens it actually holds. A value
        near 1.0 means blocks are packed; contiguous allocation (sized to max_len)
        sits far below it on mixed-length workloads.
        """
        allocated = sum(len(t) for t in self._tables.values()) * self.block_size
        if allocated == 0:
            return 0.0
        return sum(live_tokens.values()) / allocated
