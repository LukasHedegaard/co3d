"""
Modules adding optimizers
"""

from math import ceil
from operator import attrgetter
from typing import Callable

import torch

from ride.core import Configs, OptimizerMixin
from ride.utils.discriminative_lr import discriminative_lr


def discounted_steps_per_epoch(
    base_steps: int, num_gpus: int, accumulate_grad_batches: int
):
    return max(
        1, ceil(base_steps / max(1, num_gpus) / max(1, accumulate_grad_batches or 1))
    )


class SgdOptimizer(OptimizerMixin):
    hparams: ...
    parameters: Callable

    def validate_attributes(self):
        attrgetter("parameters")(self)
        for hparam in SgdOptimizer.configs().names:
            attrgetter(f"hparams.{hparam}")(self)

    @staticmethod
    def configs() -> Configs:
        c = Configs()
        c.add(
            name="learning_rate",
            type=float,
            default=0.1,
            choices=(5e-2, 5e-1),
            strategy="loguniform",
            description="Learning rate.",
        )
        c.add(
            name="weight_decay",
            type=float,
            default=1e-5,
            choices=(1e-6, 1e-3),
            strategy="loguniform",
            description="Weight decay.",
        )
        c.add(
            name="momentum",
            type=float,
            default=0.9,
            choices=(0, 0.999),
            strategy="uniform",
            description="Momentum.",
        )
        return c

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


class AdamWOptimizer(OptimizerMixin):
    hparams: ...
    parameters: Callable

    def validate_attributes(self):
        attrgetter("parameters")(self)
        for hparam in AdamWOptimizer.configs().names:
            attrgetter(f"hparams.{hparam}")(self)

    @staticmethod
    def configs() -> Configs:
        c = Configs()
        c.add(
            name="learning_rate",
            type=float,
            default=0.001,
            choices=(5e-7, 5e-1),
            strategy="loguniform",
            description="Learning rate.",
        )
        c.add(
            name="optimizer_beta1",
            type=float,
            default=0.9,
            choices=(0, 0.999),
            strategy="uniform",
            description="Beta 1.",
        )
        c.add(
            name="optimizer_beta2",
            type=float,
            default=0.999,
            choices=(0, 0.99999),
            strategy="uniform",
            description="Beta 2.",
        )
        c.add(
            name="weight_decay",
            type=float,
            default=1e-2,
            choices=(1e-6, 1e-1),
            strategy="loguniform",
            description="Weight decay.",
        )
        return c

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(self.hparams.optimizer_beta1, self.hparams.optimizer_beta2),
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


class SgdReduceLrOnPlateauOptimizer(OptimizerMixin):
    hparams: ...
    parameters: Callable

    def validate_attributes(self):
        attrgetter("parameters")(self)
        for hparam in SgdReduceLrOnPlateauOptimizer.configs().names:
            attrgetter(f"hparams.{hparam}")(self)

    @staticmethod
    def configs() -> Configs:
        c = SgdOptimizer.configs()
        c.add(
            name="learning_rate_reduction_patience",
            type=int,
            default=10,
            strategy="choice",
            description="Number of epochs before learning rate should be reduced",
        )
        c.add(
            name="learning_rate_reduction_factor",
            type=float,
            default=0.1,
            strategy="choice",
            description="Reduction factor when learning rate is reduced",
        )
        return c

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            patience=self.hparams.learning_rate_reduction_patience,
            factor=self.hparams.learning_rate_reduction_factor,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train/loss",
        }


class AdamWReduceLrOnPlateauOptimizer(OptimizerMixin):
    hparams: ...
    parameters: Callable

    def validate_attributes(self):
        attrgetter("parameters")(self)
        for hparam in AdamWReduceLrOnPlateauOptimizer.configs().names:
            attrgetter(f"hparams.{hparam}")(self)

    @staticmethod
    def configs() -> Configs:
        c = AdamWOptimizer.configs()
        c.add(
            name="learning_rate_reduction_patience",
            type=int,
            default=10,
            strategy="choice",
            description="Number of epochs before learning rate should be reduced",
        )
        c.add(
            name="learning_rate_reduction_factor",
            type=float,
            default=0.1,
            strategy="choice",
            description="Reduction factor when learning rate is reduced",
        )
        return c

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(self.hparams.optimizer_beta1, self.hparams.optimizer_beta2),
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            patience=self.hparams.learning_rate_reduction_patience,
            factor=self.hparams.learning_rate_reduction_factor,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train/loss",
        }


class SgdCyclicLrOptimizer(OptimizerMixin):
    hparams: ...
    parameters: Callable
    train_dataloader: Callable

    def validate_attributes(self):
        attrgetter("parameters")(self)
        attrgetter("train_dataloader")(self)
        attrgetter("hparams.batch_size")(self)
        attrgetter("hparams.num_gpus")(self)
        attrgetter("hparams.max_epochs")(self)
        attrgetter("hparams.accumulate_grad_batches")(self)
        for hparam in SgdCyclicLrOptimizer.configs().names:
            attrgetter(f"hparams.{hparam}")(self)

    @staticmethod
    def configs() -> Configs:
        c = SgdOptimizer.configs()
        c.add(
            name="discriminative_lr_fraction",
            type=float,
            default=1,
            choices=(1e-7, 1),
            strategy="loguniform",
            description=(
                "Discriminative learning rate fraction of early layers compared to final layers. "
                "If `1`, discriminative learning rate is not used."
            ),
        )
        return c

    def configure_optimizers(self):
        params, lr = discriminative_lr_and_params(
            self, self.hparams.learning_rate, self.hparams.discriminative_lr_fraction
        )
        optimizer = torch.optim.SGD(
            params=params,
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        # Use recommendations from https://arxiv.org/abs/1506.01186
        base_lr = [x / 4 for x in lr] if isinstance(lr, list) else lr / 4
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=lr,
            step_size_up=discounted_steps_per_epoch(
                len(self.train_dataloader()) / 4,
                self.hparams.num_gpus,
                self.hparams.accumulate_grad_batches,
            ),
            step_size_down=discounted_steps_per_epoch(
                (len(self.train_dataloader()) - len(self.train_dataloader()) / 4),
                self.hparams.num_gpus,
                self.hparams.accumulate_grad_batches,
            ),
            cycle_momentum=True,  # Not supported
        )
        return [optimizer], {"scheduler": scheduler, "interval": "step"}


class AdamWCyclicLrOptimizer(OptimizerMixin):
    hparams: ...
    parameters: Callable
    train_dataloader: Callable

    def validate_attributes(self):
        attrgetter("parameters")(self)
        attrgetter("train_dataloader")(self)
        attrgetter("hparams.batch_size")(self)
        attrgetter("hparams.max_epochs")(self)
        attrgetter("hparams.accumulate_grad_batches")(self)
        for hparam in AdamWCyclicLrOptimizer.configs().names:
            attrgetter(f"hparams.{hparam}")(self)

    @staticmethod
    def configs() -> Configs:
        c = AdamWOptimizer.configs()
        c.add(
            name="discriminative_lr_fraction",
            type=float,
            default=1,
            choices=(1e-7, 1),
            strategy="loguniform",
            description=(
                "Discriminative learning rate fraction of early layers compared to final layers. "
                "If `1`, discriminative learning rate is not used."
            ),
        )
        return c

    def configure_optimizers(self):
        params, lr = discriminative_lr_and_params(
            self, self.hparams.learning_rate, self.hparams.discriminative_lr_fraction
        )
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.learning_rate,
            betas=(self.hparams.optimizer_beta1, self.hparams.optimizer_beta2),
            weight_decay=self.hparams.weight_decay,
        )
        # Use recommendations from https://arxiv.org/abs/1506.01186
        base_lr = [x / 4 for x in lr] if isinstance(lr, list) else lr / 4
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=lr,
            step_size_up=discounted_steps_per_epoch(
                len(self.train_dataloader()) / 4,
                self.hparams.num_gpus,
                self.hparams.accumulate_grad_batches,
            ),
            step_size_down=discounted_steps_per_epoch(
                len(self.train_dataloader()) - len(self.train_dataloader()) / 4,
                self.hparams.num_gpus,
                self.hparams.accumulate_grad_batches,
            ),
            cycle_momentum=False,
        )
        return [optimizer], {"scheduler": scheduler, "interval": "step"}


class SgdOneCycleOptimizer(OptimizerMixin):
    hparams: ...
    parameters: Callable
    train_dataloader: Callable

    def validate_attributes(self):
        attrgetter("parameters")(self)
        attrgetter("train_dataloader")(self)
        attrgetter("hparams.max_epochs")(self)
        attrgetter("hparams.batch_size")(self)
        attrgetter("hparams.num_gpus")(self)
        attrgetter("hparams.accumulate_grad_batches")(self)
        for hparam in SgdOneCycleOptimizer.configs().names:
            attrgetter(f"hparams.{hparam}")(self)

    @staticmethod
    def configs() -> Configs:
        c = SgdOptimizer.configs()
        c.add(
            name="discriminative_lr_fraction",
            type=float,
            default=1,
            choices=(1e-7, 1),
            strategy="loguniform",
            description=(
                "Discriminative learning rate fraction of early layers compared to final layers. "
                "If `1`, discriminative learning rate is not used."
            ),
        )
        return c

    def configure_optimizers(self):
        params, lr = discriminative_lr_and_params(
            self, self.hparams.learning_rate, self.hparams.discriminative_lr_fraction
        )
        optimizer = torch.optim.SGD(
            params=params,
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=discounted_steps_per_epoch(
                len(self.train_dataloader()),
                self.hparams.num_gpus,
                self.hparams.accumulate_grad_batches,
            ),
            epochs=self.hparams.max_epochs,
        )
        return [optimizer], {"scheduler": scheduler, "interval": "step"}


class AdamWOneCycleOptimizer(OptimizerMixin):
    hparams: ...
    parameters: Callable
    train_dataloader: Callable

    def validate_attributes(self):
        attrgetter("parameters")(self)
        attrgetter("train_dataloader")(self)
        attrgetter("hparams.batch_size")(self)
        attrgetter("hparams.max_epochs")(self)
        attrgetter("hparams.num_gpus")(self)
        attrgetter("hparams.accumulate_grad_batches")(self)
        for hparam in AdamWOneCycleOptimizer.configs().names:
            attrgetter(f"hparams.{hparam}")(self)

    @staticmethod
    def configs() -> Configs:
        c = AdamWOptimizer.configs()
        c.add(
            name="discriminative_lr_fraction",
            type=float,
            default=1,
            choices=(1e-7, 1),
            strategy="loguniform",
            description=(
                "Discriminative learning rate fraction of early layers compared to final layers. "
                "If `1`, discriminative learning rate is not used."
            ),
        )
        return c

    def configure_optimizers(self):
        params, lr = discriminative_lr_and_params(
            self, self.hparams.learning_rate, self.hparams.discriminative_lr_fraction
        )
        optimizer = torch.optim.AdamW(
            params=params,
            lr=self.hparams.learning_rate,
            betas=(self.hparams.optimizer_beta1, self.hparams.optimizer_beta2),
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=discounted_steps_per_epoch(
                len(self.train_dataloader()),
                self.hparams.num_gpus,
                self.hparams.accumulate_grad_batches,
            ),
            epochs=self.hparams.max_epochs,
        )
        return [optimizer], {"scheduler": scheduler, "interval": "step"}


def discriminative_lr_and_params(
    model: torch.nn.Module, lr: float, discriminative_lr_fraction: float
):
    if discriminative_lr_fraction != 1:
        params, max_lr, _ = discriminative_lr(
            model,
            slice(
                lr * discriminative_lr_fraction,
                lr,
            ),
        )
    else:
        params = model.parameters()
        max_lr = lr

    return params, max_lr
