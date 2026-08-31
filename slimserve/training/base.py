"""Base trainer — Template Method pattern (Phase 3).

The training *lifecycle* is fixed — load the model, prepare the dataset, build the
underlying (library) trainer, run it, save a mergeable checkpoint — while the
*how* of each step varies. Subclasses fill the hooks.

Why a library-owned loop instead of a hand-rolled one: Unsloth/TRL already provide
a battle-tested, memory-optimized training loop; re-implementing it would be
strictly worse. So the varying step here is **which trainer object we build**
(``build_trainer``), not a hand-written ``compute_loss``. A future logit-KD trainer
reuses this exact lifecycle and overrides only ``build_trainer`` to return a
KD-loss trainer — that's OCP: new behavior via a new subclass, no edits here.
"""
from __future__ import annotations

from abc import abstractmethod

from slimserve.core.config import TrainConfig
from slimserve.core.interfaces import Trainer
from slimserve.training.checkpoints import latest_checkpoint


class BaseTrainer(Trainer):
    def train(self, config: TrainConfig) -> str:
        model, tokenizer = self.load_model(config)
        dataset = self.prepare_dataset(config, tokenizer)
        trainer = self.build_trainer(model, tokenizer, dataset, config)
        # Auto-resume: if a prior run left checkpoints in output_dir, continue from
        # the newest one. Re-running the same command after a kill just picks up.
        resume = latest_checkpoint(config.output_dir)
        if resume:
            print(f"resuming from {resume}")
        trainer.train(resume_from_checkpoint=resume)
        return self.save(model, tokenizer, config)

    @abstractmethod
    def load_model(self, config: TrainConfig):
        """Return (model, tokenizer), e.g. a 4-bit base with LoRA adapters."""
        ...

    @abstractmethod
    def prepare_dataset(self, config: TrainConfig, tokenizer):
        """Return a training dataset the underlying trainer accepts."""
        ...

    @abstractmethod
    def build_trainer(self, model, tokenizer, dataset, config: TrainConfig):
        """Return an object with a ``.train()`` method (e.g. a TRL SFTTrainer)."""
        ...

    @abstractmethod
    def save(self, model, tokenizer, config: TrainConfig) -> str:
        """Persist a standalone checkpoint; return its path."""
        ...
