"""Continuous-batching scheduler (Phase 3).

The idea that makes real serving fast: instead of running one batch of requests to
completion before starting the next, keep a *running set* that's refilled every
decode step. A sequence that hits EOS leaves immediately and a waiting request
takes its slot the same step — so short and long requests share the GPU without
the short ones waiting behind the long ones (no head-of-line blocking).

This class is pure queue bookkeeping (no tensors), so it unit-tests on CPU. The
engine calls ``admit`` per request, ``next_batch`` each step to get the sequences
to advance, and ``retire`` when one finishes.
"""
from __future__ import annotations

from collections import deque

from slimserve.core.config import GenerationRequest
from slimserve.core.interfaces import Scheduler
from slimserve.core.registry import register


@register("scheduler", "continuous")
class ContinuousBatchScheduler(Scheduler):
    def __init__(self, max_num_seqs: int) -> None:
        self.max_num_seqs = max_num_seqs
        self._next_id = 0
        self._waiting: deque[int] = deque()      # admitted, not yet running
        self._running: list[int] = []            # advancing this and each step
        self._requests: dict[int, GenerationRequest] = {}

    def admit(self, request: GenerationRequest) -> int:
        """Register a request; assign and return its seq_id (queued as waiting)."""
        seq_id = self._next_id
        self._next_id += 1
        self._requests[seq_id] = request
        self._waiting.append(seq_id)
        return seq_id

    def next_batch(self) -> list[int]:
        """Promote waiting sequences into the running set up to ``max_num_seqs``,
        then return the running seq_ids to advance this decode step."""
        while self._waiting and len(self._running) < self.max_num_seqs:
            self._running.append(self._waiting.popleft())
        return list(self._running)

    def retire(self, seq_id: int) -> None:
        """Remove a finished sequence (EOS or max_tokens); frees its running slot so
        ``next_batch`` admits a waiter next step. Not on the ABC — engine-scheduler
        collaboration stays concrete (ISP: keep the interface tiny)."""
        if seq_id in self._running:
            self._running.remove(seq_id)
        self._requests.pop(seq_id, None)

    def request(self, seq_id: int) -> GenerationRequest:
        return self._requests[seq_id]

    def has_work(self) -> bool:
        return bool(self._running or self._waiting)
