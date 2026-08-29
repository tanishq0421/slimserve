"""QLoRA fine-tuning via transformers + peft + bitsandbytes (Phase 3).

Deliberately NOT Unsloth: Unsloth drags in vLLM, which repeatedly broke the
Kaggle environment (CUDA-lib mismatch). This portable stack — 4-bit base
(bitsandbytes NF4) + LoRA adapters (peft) + the plain HuggingFace Trainer — has
no vLLM dependency and is version-stable.

One trainer handles both recipes (gold SFT and sequence distillation); only the
dataset file differs. Heavy imports live inside the hooks so the package imports
fine on a GPU-less machine.
"""
from __future__ import annotations

from slimserve.core.config import TrainConfig
from slimserve.core.registry import register
from slimserve.training.base import BaseTrainer
from slimserve.training.dataset import load_records, to_dataset

_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]


@register("trainer", "qlora")
class QLoRATrainer(BaseTrainer):
    def load_model(self, config: TrainConfig):
        import torch
        from peft import (LoraConfig, get_peft_model,
                          prepare_model_for_kbit_training)
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   BitsAndBytesConfig)

        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quant = None
        if config.load_in_4bit:                     # the "Q" in QLoRA
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=quant,
            torch_dtype=torch.float16,
            device_map={"": 0},
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(   # the "LoRA" adapters
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=_LORA_TARGETS,
        ))
        model.print_trainable_parameters()
        return model, tokenizer

    def prepare_dataset(self, config: TrainConfig, tokenizer):
        max_len = config.extra.get("max_seq_len", 2048)
        return to_dataset(load_records(config.dataset), tokenizer, max_len)

    def build_trainer(self, model, tokenizer, dataset, config: TrainConfig):
        from transformers import (DataCollatorForSeq2Seq, Trainer,
                                   TrainingArguments)

        args = TrainingArguments(
            output_dir=config.output_dir,
            per_device_train_batch_size=config.extra.get("batch_size", 8),
            gradient_accumulation_steps=config.extra.get("grad_accum", 2),
            num_train_epochs=config.epochs,
            learning_rate=config.lr,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
            logging_steps=10,
            fp16=True,                              # T4 has no bf16
            gradient_checkpointing=True,            # trade compute for memory
            gradient_checkpointing_kwargs={"use_reentrant": False},
            seed=config.extra.get("seed", 3407),
            report_to="none",
            save_strategy="no",
        )
        collator = DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100)
        return Trainer(model=model, args=args, train_dataset=dataset,
                       data_collator=collator)

    def save(self, model, tokenizer, config: TrainConfig) -> str:
        # You can't cleanly merge LoRA into a 4-bit base, so: save the adapter,
        # reload a fresh fp16 base on CPU, merge, and write a standalone
        # checkpoint that our vLLM engine can serve directly in Step 5.
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        adapter_dir = f"{config.output_dir}_adapter"
        model.save_pretrained(adapter_dir)

        base = AutoModelForCausalLM.from_pretrained(
            config.base_model, torch_dtype=torch.float16)   # CPU, avoids GPU OOM
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        merged.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        return config.output_dir
