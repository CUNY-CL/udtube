"""Custom callbacks."""

import logging
import sys
from typing import Iterator, Optional, Sequence, TextIO

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
    """

    path: Optional[str]
    sink: TextIO
    mapper: data.Mapper

    def __init__(
        self,
        path: Optional[str] = None,  # If not filled in, stdout will be used.
    ):
        super().__init__("batch")
        self.path = path
        self.sink = sys.stdout

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
        mapper = data.Mapper(trainer.datamodule.index)
        # Batch-level argmax on the classification heads.
        upos_hat = (
            torch.argmax(logits.upos, dim=1) if logits.use_upos else None
        )
        xpos_hat = (
            torch.argmax(logits.xpos, dim=1) if logits.use_xpos else None
        )
        lemma_hat = (
            torch.argmax(logits.lemma, dim=1) if logits.use_lemma else None
        )
        feats_hat = (
            torch.argmax(logits.feats, dim=1) if logits.use_feats else None
        )
        for i, tokenlist in enumerate(batch.tokenlists):
            # Sentence-level decoding of the classification indices, followed
            # by rewriting the fields in the tokenlist.
            if upos_hat is not None:
                upos_it = mapper.decode_upos(upos_hat[i, :])
                self._fill_in_tags(tokenlist, "upos", upos_it)
            if xpos_hat is not None:
                xpos_it = mapper.decode_xpos(xpos_hat[i, :])
                self._fill_in_tags(tokenlist, "xpos", xpos_it)
            if lemma_hat is not None:
                lemma_it = mapper.decode_lemma(
                    tokenlist.get_tokens(), lemma_hat[i, :]
                )
                self._fill_in_tags(tokenlist, "lemma", lemma_it)
            if feats_hat is not None:
                feats_it = mapper.decode_feats(feats_hat[i, :])
                self._fill_in_tags(tokenlist, "feats", feats_it)
            print(tokenlist, file=self.sink)
        self.sink.flush()

    @staticmethod
    def _fill_in_tags(
        tokenlist: data.conllu.TokenList, attr: str, tags: Iterator[str]
    ) -> None:
        """Helper method for copying tags into tokenlist.

        Args:
            tokenlist (data.conllu.TokenList): tokenlist to insert into.
            attr (str): attribute on tokens where the tags should be inserted.
            tags (Iterator[str]): tags to insert.
        """
        # Note that when MWEs are present, the iterators with predicted tags
        # from the classifier heads are shorter than the tokenlists, so we
        # `continue` without advancing said iterators.
        for token in tokenlist:
            if token.is_mwe:
                continue
            try:
                setattr(token, attr, next(tags))
            except StopIteration:
                # Prevents the error from being caught by Lightning.
                logging.error(
                    f"Length mismatch at tag {attr!r} (sent_id: "
                    f"{tokenlist.metadata.get('sent_id')})"
                )
                continue

    def on_predict_end(
        self, trainer: trainer.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        if self.sink is not sys.stdout:
            self.sink.close()
