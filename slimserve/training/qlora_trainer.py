"""QLoRA fine-tuning via Unsloth (Phase 3).

One trainer handles BOTH recipes — plain SFT-on-gold and sequence-level
distillation — because sequence KD *is* just SFT on the teacher's completions.
The only thing that differs is which dataset file ``config.dataset`` points at
(built in Phase 2). Logit KD, which needs a custom loss, is a separate trainer.

Heavy imports (unsloth/trl) live inside the hooks so the package imports fine on a
machine without a GPU; the training path itself is exercised on Kaggle.
"""
from __future__ import annotations

from slimserve.core.config import TrainConfig
from slimserve.core.registry import register
from slimserve.training.base import BaseTrainer
from slimserve.training.dataset import build_dataset, load_records

# Qwen2.5 chat (ChatML) turn markers — used to train on the assistant turn only.
_USER_MARK = "<|im_start|>user\n"
_ASSISTANT_MARK = "<|im_start|>assistant\n"
_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]


@register("trainer", "qlora")
class QLoRATrainer(BaseTrainer):
    def load_model(self, config: TrainConfig):
        from unsloth import FastLanguageModel

        max_seq = config.extra.get("max_seq_len", 2048)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.base_model,
            max_seq_length=max_seq,
            load_in_4bit=config.load_in_4bit,      # the "Q" in QLoRA (NF4 base)
            dtype=None,
        )
        model = FastLanguageModel.get_peft_model(   # the "LoRA" adapters
            model,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=0.0,
            target_modules=_LORA_TARGETS,
            use_gradient_checkpointing="unsloth",
            random_state=config.extra.get("seed", 3407),
        )
        return model, tokenizer

    def prepare_dataset(self, config: TrainConfig, tokenizer):
        return build_dataset(load_records(config.dataset), tokenizer)

    def build_trainer(self, model, tokenizer, dataset, config: TrainConfig):
        from trl import SFTConfig, SFTTrainer
        from unsloth.chat_templates import train_on_responses_only

        args = SFTConfig(
            output_dir=config.output_dir,
            per_device_train_batch_size=config.extra.get("batch_size", 8),
            gradient_accumulation_steps=config.extra.get("grad_accum", 2),
            num_train_epochs=config.epochs,
            learning_rate=config.lr,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            logging_steps=10,
            seed=config.extra.get("seed", 3407),
            report_to="none",
            dataset_text_field="text",
            max_seq_length=config.extra.get("max_seq_len", 2048),
        )
        trainer = SFTTrainer(model=model, tokenizer=tokenizer,
                             train_dataset=dataset, args=args)
        # completion-only loss: mask the prompt, train only on the assistant turn.
        return train_on_responses_only(
            trainer, instruction_part=_USER_MARK, response_part=_ASSISTANT_MARK)

    def save(self, model, tokenizer, config: TrainConfig) -> str:
        # merge LoRA into the base and save a standalone 16-bit checkpoint that
        # our vLLM engine can serve directly in Step 5.
        model.save_pretrained_merged(
            config.output_dir, tokenizer, save_method="merged_16bit")
        return config.output_dir
