"""Custom schedulers."""

import math

from torch import optim


class WarmupInverseSquareRoot(optim.lr_scheduler.LambdaLR):
    """Linear warmup and then inverse square root decay.

    Linearly increases learning rate from 0 to the learning rate over the
    warmup epochs, then decreases learning rate according to an inverse root
    square schedule.

    A slightly different version is present in Yoyodyne.

    After:
        Wu, S., Cotterell, R., and Hulden, M. 2021. Applying the transformer to
        character-level transductions. In Proceedings of the 16th Conference of
        the European Chapter of the Association for Computational Linguistics:
        Main Volume, pages 1901-1907.

    Args:
        optimizer: optimizer.
        warmup_epochs: number of warmup epochs.
    """

    warmup_epochs: int
    decay_factor: float

    def __init__(
        self,
        optimizer,
        warmup_epochs,
    ):
        self.warmup_epochs = warmup_epochs
        self.decay_factor = math.sqrt(warmup_epochs)
        super().__init__(optimizer, self.lr_lambda)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.optimizer}, {self.warmup_epochs})"
        )

    def lr_lambda(self, epoch: int) -> float:
        """Computes the learning rate lambda at a given epoch.

         Args:
            epoch: current epoch.

        Returns:
            float: lr_lambda.
        """
        if self.warmup_epochs < 1:
            return self.decay_factor
        if epoch < self.warmup_epochs:
            return float(epoch) / float(max(1, self.warmup_epochs))
        return self.decay_factor * epoch**-0.5
