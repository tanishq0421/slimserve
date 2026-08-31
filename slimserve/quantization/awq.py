"""AWQ (Activation-aware Weight Quantization) — Phase 1, Week 2.

Strategy implementation of the Quantizer interface. AWQ protects the small
fraction of salient weight channels (those multiplying large-magnitude
activations) so INT4 barely dents quality — that's the intuition to be able to
explain in an interview.
"""
from __future__ import annotations

from slimserve.core.config import Precision, QuantConfig
from slimserve.core.interfaces import Quantizer
from slimserve.core.registry import register


@register("quantizer", "awq")
class AWQQuantizer(Quantizer):
    def quantize(self, model_path: str, config: QuantConfig) -> str:
        """Quantize an fp16 checkpoint (local dir or HF id) to AWQ and save it.

        Returns the output path, which vLLM then serves with ``quantization: awq``.
        Uses AWQ's default calibration set; for a task-specialized model, calibrating
        on in-domain prompts could help, but the default is the robust baseline.
        """
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        w_bit = 8 if config.precision is Precision.INT8 else 4
        quant_config = {
            "zero_point": True,
            "q_group_size": int(config.extra.get("group_size", 128)),
            "w_bit": w_bit,
            "version": "GEMM",
        }
        model = AutoAWQForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.quantize(tokenizer, quant_config=quant_config)   # default calibration
        out_path = config.out_path or f"{model_path.rstrip('/')}-awq"
        model.save_quantized(out_path)
        tokenizer.save_pretrained(out_path)
        return out_path
