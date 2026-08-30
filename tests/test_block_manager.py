"""CPU unit tests for the paged-KV block manager — pure logic, no model, no GPU."""
import pytest

from slimserve.engines.mini_engine.block_manager import BlockManager


def test_allocate_reserves_ceil_blocks():
    bm = BlockManager(block_size=4, num_blocks=10)
    bm.allocate(seq_id=1, num_tokens=6)          # ceil(6/4) = 2 blocks
    assert len(bm.block_table(1)) == 2
    assert bm.num_free_blocks() == 8


def test_position_maps_to_block_and_offset():
    bm = BlockManager(block_size=4, num_blocks=10)
    bm.allocate(1, num_tokens=8)                  # 2 blocks, positions 0..7
    table = bm.block_table(1)
    assert bm.append_slot(1, 0) == (table[0], 0)
    assert bm.append_slot(1, 5) == (table[1], 1)  # 5 // 4 = block 1, 5 % 4 = offset 1


def test_append_grows_table_across_block_boundary():
    bm = BlockManager(block_size=4, num_blocks=10)
    bm.allocate(1, num_tokens=4)                  # 1 block, positions 0..3
    assert len(bm.block_table(1)) == 1
    block, offset = bm.append_slot(1, 4)          # position 4 needs a 2nd block
    assert offset == 0 and len(bm.block_table(1)) == 2
    assert bm.num_free_blocks() == 8              # one for allocate, one for the grow


def test_free_returns_blocks_to_pool():
    bm = BlockManager(block_size=4, num_blocks=10)
    bm.allocate(1, num_tokens=8)
    assert bm.num_free_blocks() == 8
    bm.free(1)
    assert bm.num_free_blocks() == 10
    with pytest.raises(KeyError):
        bm.block_table(1)                         # gone after free


def test_out_of_blocks_raises():
    bm = BlockManager(block_size=4, num_blocks=2)
    assert bm.can_allocate(8)                     # ceil(8/4) = 2, exactly fits
    bm.allocate(1, num_tokens=8)
    assert not bm.can_allocate(1)
    with pytest.raises(MemoryError):
        bm.allocate(2, num_tokens=1)


def test_utilization_reflects_packing():
    bm = BlockManager(block_size=4, num_blocks=10)
    bm.allocate(1, num_tokens=4)                  # 1 block = 4 slots
    bm.allocate(2, num_tokens=4)                  # 1 block = 4 slots; 8 slots total
    # seq 1 holds 4 tokens (full block), seq 2 holds 1 token (mostly empty block)
    assert bm.utilization({1: 4, 2: 1}) == pytest.approx(5 / 8)
    assert BlockManager(4, 10).utilization({}) == 0.0
