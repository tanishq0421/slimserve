"""AWQ (Activation-aware Weight Quantization) — Phase 1, Week 2.

Strategy implementation of the Quantizer interface. AWQ protects the small
fraction of salient weight channels (those multiplying large-magnitude
activations) so INT4 barely dents quality — that's the intuition to be able to
explain in an interview.
"""
from __future__ import annotations

from slimserve.core.config import QuantConfig
from slimserve.core.interfaces import Quantizer
from slimserve.core.registry import register


@register("quantizer", "awq")
class AWQQuantizer(Quantizer):
    def quantize(self, model_path: str, config: QuantConfig) -> str:
        # from awq import AutoAWQForCausalLM
        # load -> model.quantize(calib_data, w_bit=4, ...) -> save
        raise NotImplementedError("Phase 1, Week 2: run AWQ, return out_path.")
