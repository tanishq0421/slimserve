"""CPU unit tests for the continuous-batching scheduler — pure queue logic."""
from slimserve.core.config import GenerationRequest
from slimserve.engines.mini_engine.scheduler import ContinuousBatchScheduler


def _req(prompt="hi"):
    return GenerationRequest(prompt=prompt)


def test_admit_assigns_incrementing_ids():
    s = ContinuousBatchScheduler(max_num_seqs=4)
    assert s.admit(_req()) == 0
    assert s.admit(_req()) == 1
    assert s.has_work()


def test_next_batch_respects_max_num_seqs():
    s = ContinuousBatchScheduler(max_num_seqs=2)
    ids = [s.admit(_req()) for _ in range(3)]
    batch = s.next_batch()
    assert batch == ids[:2]                       # third stays waiting
    assert s.next_batch() == ids[:2]              # still full, no new promotion


def test_retire_frees_slot_for_waiting_request():
    s = ContinuousBatchScheduler(max_num_seqs=2)
    a, b, c = (s.admit(_req()) for _ in range(3))
    assert s.next_batch() == [a, b]
    s.retire(a)                                   # a finishes
    assert s.next_batch() == [b, c]               # c promoted into the freed slot


def test_has_work_false_when_drained():
    s = ContinuousBatchScheduler(max_num_seqs=2)
    a = s.admit(_req())
    s.next_batch()
    s.retire(a)
    assert not s.has_work()


def test_request_is_retrievable_then_gone():
    s = ContinuousBatchScheduler(max_num_seqs=2)
    a = s.admit(_req(prompt="weather in Paris"))
    assert s.request(a).prompt == "weather in Paris"
    s.retire(a)
    assert a not in s._requests
