"""Models."""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import nn, optim
from torchmetrics.functional import classification
import transformers

from . import data, defaults, encoders, special


class UDTube(pl.LightningModule):
    """UDTube model.

    This model handles POS tagging, lemmatization, and morphological feature
    classification using a single shared, pre-trained, tuned BERT-style
    encoder.

    Args:
        model_name: The name of the model; used to tokenize and encode.
        encoder_dropout: Dropout probability for the encoder layers.
        encoder_learning_rate: Learning rate for the encoder layers.
        pooling_layers: Number of encoder layers to use to compute the
            embedding.
        classifier_dropout: Dropout probability for the classifier layers.
        classifier_learning_rate: Learning rate for the classifier layers.
        warmup_steps: Number of warmup steps.
        use_upos: If true, use POS tags.
        use_xpos: If true, use language-specific POS tags.
        use_lemma: If true, use lemmatization.
        use_feats: If true, use morphological feature tags.
        upos_out_size: Number of POS labels; usually provided by the
            dataset object.
        xpos_out_size: Number of language-specific POS labels; usually
            provided by the dataset object.
        lemma_out_size: Number of lemma tags; usually provided by the
            dataset.
        feats_out_size: Number of feature tags; usually provided by the
            dataset.
    """

    # TODO: declare instance types.

    # Initialization.

    def _make_head(self, out_size: int) -> nn.Sequential:
        """Helper for generating heads.

        Args:
            out_size (int).

        Returns:
            A sequential linear layer.
        """
        return nn.Sequential(
            nn.Linear(self.hidden_size, out_size), nn.LeakyReLU()
        )

    def __init__(
        self,
        model_name: str = defaults.ENCODER,
        encoder_dropout: float = defaults.ENCODER_DROPOUT,
        encoder_learning_rate: float = defaults.ENCODER_LEARNING_RATE,
        pooling_layers: int = defaults.POOLING_LAYERS,
        classifier_dropout: float = defaults.CLASSIFIER_DROPOUT,
        classifier_learning_rate: float = defaults.CLASSIFIER_LEARNING_RATE,
        warmup_steps: int = defaults.WARMUP_STEPS,
        use_upos: bool = defaults.USE_POS,
        use_xpos: bool = defaults.USE_XPOS,
        use_lemma: bool = defaults.USE_LEMMA,
        use_feats: bool = defaults.USE_FEATS,
        # `2` is a dummy value here; it will be set by the data set object.
        upos_out_size: int = 2,
        xpos_out_size: int = 2,
        lemma_out_size: int = 2,
        feats_out_size: int = 2,
    ):
        super().__init__()
        self.encoder = encoders.load(model_name, encoder_dropout)
        self.encoder_learning_rate = encoder_learning_rate
        self.pooling_layers = pooling_layers
        # Freezes encoder layer params for the first epoch.
        for p in self.encoder.parameters():
            p.requires_grad = False
        logging.info("Encoder parameters frozen for the first epoch")
        self.upos_head = self._make_head(upos_out_size) if use_upos else None
        self.xpos_head = self._make_head(xpos_out_size) if use_xpos else None
        self.lemma_head = (
            self._make_head(lemma_out_size) if use_lemma else None
        )
        self.feats_head = (
            self._make_head(feats_out_size) if use_feats else None
        )
        # Essentially, this is dropout for the context-depending encodings
        # going into the classifier.
        self.dropout_layer = nn.Dropout(classifier_dropout)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=special.PAD_IDX)
        self.warmup_steps = warmup_steps
        self.step_adjustment = None
        self.save_hyperparameters()

    # Properties.

    @property
    def hidden_size(self) -> int:
        return self.encoder.config.hidden_size

    @property
    def use_upos(self) -> bool:
        return self.upos_head is not None

    @property
    def use_xpos(self) -> bool:
        return self.xpos_head is not None

    @property
    def use_lemma(self) -> bool:
        return self.lemma_head is not None

    @property
    def use_feats(self) -> bool:
        return self.feats_head is not None

    # Forward pass.

    def _group_embeddings(
        self,
        embeddings: torch.Tensor,
        tokenized: transformers.BatchEncoding,
    ) -> Tuple[torch.Tensor, List[List[str]]]:
        """Groups subword embeddings to form word embeddings.

        This is necessary because each classifier head makes per-word
        decisions, but the contextual embeddings use subwords. Therefore,
        we average over subwords.

        Args:
            embeddings: the embeddings tensor to pool.
            tokens: the batch encoding.

        Returns:
            The re-pooled embeddings tensor.
        """
        embeddings_list = []
        for encoding, x_embed_i in zip(tokenized.encodings, embeddings):
            embeddings_i = []
            last_word_idx = slice(0, 0)
            for word_id, x_embed_j in zip(encoding.word_ids, x_embed_i):
                # Padding.
                if word_id is None:
                    break
                start, end = encoding.word_to_tokens(word_id)
                word_idxs = slice(start, end)
                if word_idxs != last_word_idx:
                    last_word_idx = word_idxs
                    embeddings_i.append(
                        # Automatically squeezes the mean dimension.
                        torch.mean(x_embed_j[word_idxs], dim=0)
                    )
            embeddings_list.append(torch.stack(embeddings_i))
        return data.pad_tensors(embeddings_list)

    # TODO: docs.
    # TODO: typing.

    def forward(
        self,
        batch: data.Batch,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        # If something is longer than an allowed sequence, we trim it down.
        actual_length = batch.tokens.input_ids.shape[1]
        max_length = self.encoder.config.max_position_embeddings
        if actual_length > max_length:
            logging.warning(
                "truncating sequence from %d to %d", actual_length, max_length
            )
            batch.tokens.input_ids = batch.tokens.input_ids[:max_length]
            batch.tokens.attention_mask = batch.tokens.attention_mask[
                :max_length
            ]
        # Gets raw embeddings.
        x = self.encoder(batch.tokens.input_ids, batch.tokens.attention_mask)
        # Stacks the pooling layers.
        x = torch.stack(x.hidden_states[-self.pooling_layers :])
        # Averages them into one embedding layer; automatically squeezes the
        # mean dimension.
        x = torch.mean(x, dim=0)
        # Applies dropout.
        x = self.dropout_layer(x)
        # Maps from subword embeddings to word-level embeddings.
        x = self._group_embeddings(x, batch.tokens)
        # Applies classification heads.
        y_upos_logits = self.upos_head(x) if self.use_upos else None
        y_xpos_logits = self.xpos_head(x) if self.use_xpos else None
        y_lemma_logits = self.lemma_head(x) if self.use_lemma else None
        y_feats_logits = self.feats_head(x) if self.use_feats else None
        # TODO(#6): make the response an object.
        return y_upos_logits, y_xpos_logits, y_lemma_logits, y_feats_logits

    # Required API.

    # TODO: add additional optimizers.

    def configure_optimizers(self):
        """Prepare optimizer and schedulers."""
        grouped_params = [
            {
                "params": self.encoder.parameters(),
                "lr": self.encoder_learning_rate,
                "weight_decay": 0.01,
            }
        ]
        if self.use_lemma:
            grouped_params.append(
                {
                    "params": self.lemma_head.parameters(),
                    "lr": self.classifier_learning_rate,
                }
            )
        if self.use_upos:
            grouped_params.append(
                {
                    "params": self.upos_head.parameters(),
                    "lr": self.classifier_learning_rate,
                }
            )
        if self.use_xpos:
            grouped_params.append(
                {
                    "params": self.xpos_head.parameters(),
                    "lr": self.classifier_learning_rate,
                }
            )
        if self.use_feats:
            grouped_params.append(
                {
                    "params": self.feats_head.parameters(),
                    "lr": self.classifier_learning_rate,
                }
            )
        optimizer = torch.optim.AdamW(grouped_params)
        return [optimizer]

    # TODO: docs.

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: optim.Optimizer,
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        optimizer.step(closure=optimizer_closure)
        if epoch > 0:
            # This only happens once params are unfrozen
            unfrozen_steps = self.trainer.global_step - self.step_adjustment
            if unfrozen_steps < self.warmup_steps:
                # Warming up LR from 0 to the encoder LR begins before the
                # encoder is unfrozen, so LR is not really at zero.
                lr_scale = 0 + float(unfrozen_steps) / self.warmup_steps
            else:
                # Decaying weight.
                lr_scale = 1 / (unfrozen_steps**0.5)
            optimizer.param_groups[0]["lr"] = (
                lr_scale * self.encoder_learning_rate
            )
            self.log(
                "encoder_lr",
                optimizer.param_groups[0]["lr"],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

    def _loss(self, golds: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Computes cross-entropy for a head.

        This handles the necessary permutation.

        Args:
            golds: gold data.
            logits: logits.

        Returns:
            A tensor containing the loss.
        """
        # TODO: explain the permutation here.
        return self.loss_func(golds, logits.permute(0, 2, 1))

    @staticmethod
    def _accuracy(golds: torch.Tensor, logits: torch.Tensor) -> float:
        """Computes multi-class micro-accuracy for a head.

        This handles the necessary permutation.

        Args:
            golds: gold data.
            logits: logits.

        Returns:
            The multi-class micro-accuracy.
        """
        return classification.multiclass_accuracy(
            # TODO: explain the permutation and transpositions here.
            logits.permute(1, 0, 2),
            golds.T,
            num_classes=logits.shape[1],
            ignore_index=special.PAD_IDX,
            average="micro",
        )

    # TODO: docs.

    def _log_accuracy(
        self,
        golds: torch.Tensor,
        logits: torch.Tensor,
        task_name: str,
        subset: str = "train",
    ) -> None:
        self.log(
            f"{subset}_{task_name}_accuracy",
            self._accuracy(golds, logits),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    # TODO: do we really want to log accuracy every step? Probably better to
    # log validation accuracy only and only at the end of an epoch.

    def training_step(
        self, batch: data.ConlluBatch, batch_idx: int, subset: str = "train"
    ) -> Dict[str, float]:
        y_upos_logits, y_xpos_logits, y_lemma_logits, y_feats_logits = self(
            batch
        )
        # Both helpers must permute the dimensions of logits. Fortunately,
        # this is a cheap operation (permutation just creates a new view).
        losses = {}
        if self.use_upos:
            losses["xpos_loss"] = self._loss(batch.upos, y_upos_logits)
            self._log_accuracy(
                batch.upos, y_upos_logits, task_name="upos", subset=subset
            )
        if self.use_xpos:
            losses["xpos_loss"] = self._loss(batch.xpos, y_xpos_logits)
            self._log_accuracy(
                batch.xpos, y_xpos_logits, task_name="xpos", subset=subset
            )
        if self.use_lemma:
            losses["lemma_loss"] = self._loss(batch.lemma, y_lemma_logits)
            self._log_accuracy(
                batch.lemmas,
                y_lemma_logits,
                task_name="lemma",
                subset=subset,
            )
        if self.use_feats:
            losses["feats_loss"] = self._loss(batch.feats, y_feats_logits)
            self._log_accuracy(
                batch.feats,
                y_feats_logits,
                task_name="feats",
                subset=subset,
            )
        # Averages the loss of all active heads.
        losses["loss"] = loss = torch.mean(torch.stack(list(losses.values())))
        self.log(
            f"{subset}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        return loss

    # TODO: docs.
    # TODO: why is this necessary?

    def on_train_epoch_end(self) -> None:
        if self.current_epoch == 0:
            # How many steps are in an epoch?
            self.step_adjustment = self.trainer.global_step
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True
            logging.info("Encoder parameters unfrozen")

    # TODO: docs.

    def validation_step(
        self, batch: data.ConlluBatch, batch_idx: int
    ) -> Dict[str, float]:
        return self.training_step(batch, batch_idx, subset="val")

    # TODO: add test step.
    # TODO: add predict step.
