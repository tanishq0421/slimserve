"""Logit-KD trainer: a QLoRA student trained against precomputed teacher logits.

Subclasses ``QLoRATrainer`` so model loading and adapter saving are reused verbatim
— only two things differ from plain SFT:
  * the dataset is the Arrow set of teacher top-k logits (``build_data --mode logits``);
  * the loss is the ``LogitKD`` strategy, injected via a custom ``Trainer.compute_loss``.

The BaseTrainer Template Method (load → prepare_dataset → build_trainer → save) is
untouched — this overrides only ``prepare_dataset`` and ``build_trainer`` (OCP).
"""
from __future__ import annotations

from slimserve.core.config import DistillConfig, TrainConfig
from slimserve.core.registry import register
from slimserve.training.distillation.logit_kd import LogitKD
from slimserve.training.qlora_trainer import QLoRATrainer, build_training_args


@register("trainer", "logit_kd")
class LogitKDTrainer(QLoRATrainer):
    def prepare_dataset(self, config: TrainConfig, tokenizer):
        from slimserve.training.dataset import load_logit_dataset

        return load_logit_dataset(config.dataset)   # columns include kd_topk_ids/vals

    def build_trainer(self, model, tokenizer, dataset, config: TrainConfig):
        from transformers import Trainer

        from slimserve.training.dataset import KDDataCollator

        strategy = LogitKD()
        distill = DistillConfig(
            strategy="logit_kd",
            teacher_model=config.extra.get("teacher_model", ""),
            temperature=float(config.extra.get("temperature", 2.0)),
            alpha=float(config.extra.get("alpha", 0.5)),
        )

        class _KDTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                # Pull the teacher tensors out before the model forward; the model
                # only sees input_ids/attention_mask/labels.
                teacher = {
                    "topk_ids": inputs.pop("kd_topk_ids"),
                    "topk_vals": inputs.pop("kd_topk_vals"),
                }
                inputs.pop("kd_mask", None)
                outputs = model(**inputs)                      # outputs.loss = CE
                loss = strategy.compute_loss(
                    {"logits": outputs.logits,
                     "labels": inputs["labels"],
                     "ce_loss": outputs.loss},
                    teacher, distill)
                return (loss, outputs) if return_outputs else loss

        args = build_training_args(config)
        return _KDTrainer(model=model, args=args, train_dataset=dataset,
                          data_collator=KDDataCollator(tokenizer))
