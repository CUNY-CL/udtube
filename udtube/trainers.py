"""Subclasses the trainer to prevent a memory leak in prediction.

The trainer's `predict` method is called with `return_predictions=True`.
Unexpectedly, this not only returns the prediction logits for each batch, but
also appends them to an ever-growing list resident in CPU memory. When
running prediction on a large file, this usually causes an out-of-memory crash.
Therefore, we subclass the trainer to override this default and then use this
as the trainer class in the CLI.
"""

from lightning.pytorch import trainer


class Trainer(trainer.Trainer):

    def predict(self, *args, **kwargs):
        return super().predict(*args, return_predictions=False, **kwargs)
