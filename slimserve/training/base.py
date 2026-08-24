"""Base trainer — Template Method pattern (Phase 3).

The training loop (data loading, optimizer step, checkpointing, logging) is the
same regardless of whether we do plain QLoRA SFT or distillation. Subclasses
override exactly one hook: ``compute_loss``.
"""
from __future__ import annotations

from abc import abstractmethod

from slimserve.core.config import TrainConfig
from slimserve.core.interfaces import Trainer


class BaseTrainer(Trainer):
    def train(self, config: TrainConfig) -> str:
        self.setup(config)
        for epoch in range(config.epochs):
            for batch in self.dataloader():
                loss = self.compute_loss(batch)   # <-- the varying step
                self.optimizer_step(loss)
            self.checkpoint(epoch)
        return config.output_dir

    # --- fixed steps (implemented once) ---
    def setup(self, config: TrainConfig) -> None:
        raise NotImplementedError("Phase 3 Wk6: load base model 4-bit + LoRA + data.")

    def dataloader(self):
        raise NotImplementedError

    def optimizer_step(self, loss) -> None:
        raise NotImplementedError

    def checkpoint(self, epoch: int) -> None:
        raise NotImplementedError

    # --- the one hook subclasses override ---
    @abstractmethod
    def compute_loss(self, batch):
        ...
