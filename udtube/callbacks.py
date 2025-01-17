"""Custom callbacks."""

import sys
from typing import Optional, Sequence, TextIO

import lightning
from lightning.pytorch import callbacks, trainer
import torch

from . import data, models


class PredictionWriter(callbacks.BasePredictionWriter):
    """Writes predictions in CoNLL-U format.

    If path is not specified, stdout is used. If using this in conjunction
    with > or |, add --trainer.enable_progress_bar false.

    Args:
        path: Path for the predictions file.
        model_dir: Path for checkpoints, indexes, and logs.
    """

    path: Optional[str]
    sink: TextIO
    mapper: data.Mapper

    def __init__(
        self,
        path: Optional[str] = None,  # If not filled in, stdout will be used.
        model_dir: str = "",  # Dummy value filled in by a link.
    ):
        super().__init__("batch")
        self.path = path
        self.sink = sys.stdout
        assert model_dir, "no model_dir specified"
        self.mapper = data.Mapper.read(model_dir)

    # Required API.

    def on_predict_start(
        self, trainer: trainer.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        # Placing this here prevents the creation of an empty file in the case
        # where a prediction callback was specified but UDTube is not running
        # in predict mode.
        if self.path:
            self.sink = open(self.path, "w")

    def write_on_batch_end(
        self,
        trainer: trainer.Trainer,
        model: models.UDTube,
        logits: data.Logits,
        batch_indices: Optional[Sequence[int]],
        batch: data.Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        y_upos_hat = (
            torch.argmax(logits.upos, dim=1) if logits.use_upos else None
        )
        y_xpos_hat = (
            torch.argmax(logits.xpos, dim=1) if logits.use_xpos else None
        )
        y_lemma_hat = (
            torch.argmax(logits.lemma, dim=1) if logits.use_lemma else None
        )
        y_feats_hat = (
            torch.argmax(logits.feats, dim=1) if logits.use_feats else None
        )
        for i, tokenlist in enumerate(batch.tokenlists):
            # Edits the relevant fields in each tokenlist.
            if y_upos_hat is not None:
                upos_hat = self.mapper.decode_upos(y_upos_hat[i, :])
                for j, upos in enumerate(upos_hat[: len(tokenlist)]):
                    tokenlist[j]["upos"] = upos
            if y_xpos_hat is not None:
                xpos_hat = self.mapper.decode_xpos(y_xpos_hat[i, :])
                for j, xpos in enumerate(xpos_hat[: len(tokenlist)]):
                    tokenlist[j]["xpos"] = xpos
            if y_lemma_hat is not None:
                lemma_hat = self.mapper.decode_lemma(
                    [token["form"] for token in tokenlist],
                    y_lemma_hat[i, :],
                )
                for j, lemma in enumerate(lemma_hat[: len(tokenlist)]):
                    tokenlist[j]["lemma"] = lemma
            if y_feats_hat is not None:
                feats_hat = self.mapper.decode_feats(y_feats_hat[i, :])
                for j, feats in enumerate(feats_hat[: len(tokenlist)]):
                    tokenlist[j]["feats"] = feats
            print(tokenlist.serialize(), file=self.sink)
        self.sink.flush()

    def on_predict_end(
        self, trainer: trainer.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        if self.sink is not sys.stdout:
            self.sink.close()
