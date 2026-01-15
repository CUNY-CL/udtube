"""The UDTube model."""

from typing import Dict, List, Optional

import lightning
from lightning.pytorch import cli
import torch
from torch import nn
from torchmetrics import classification

from . import data, defaults, metrics, modules, special


class UDTube(lightning.LightningModule):
    """UDTube model.

    This model handles POS tagging, lemmatization, and morphological feature
    tagging using a single shared, pre-trained, fine-tuned BERT-style encoder
    and sequential linear classifiers for each subtask.

    Args:
        dropout: Dropout probability.
        encoder: Name of the Hugging Face model used to tokenize and encode.
        pooling_layers: Number of layers to use to compute the embedding.
        use_upos: Enables the universal POS tagging task.
        use_xpos: Enables the language-specific POS tagging task.
        use_lemma: Enables the lemmatization task.
        use_feats: Enables the morphological feature tagging task.
        use_parse: Enables the dependenchy parsing task.
        arc_mlp_size: Size of the arc MLP for dependency parsing.
        label_mlp_size: Size of the label MLP for dependency parsing.
    """

    encoder: modules.UDTubeEncoder
    classifier: modules.UDTubeClassifier
    loss_func: nn.CrossEntropyLoss
    # Used for validation in `fit` and testing in `test`.
    upos_accuracy: Optional[classification.MulticlassAccuracy]
    xpos_accuracy: Optional[classification.MulticlassAccuracy]
    lemma_accuracy: Optional[classification.MulticlassAccuracy]
    feats_accuracy: Optional[classification.MulticlassAccuracy]
    uas: Optional[metrics.UnlabeledAttachmentScore]
    las: Optional[metrics.LabeledAttachmentScore]

    def __init__(
        self,
        *,
        dropout: float = defaults.DROPOUT,
        encoder: str = defaults.ENCODER,
        pooling_layers: int = defaults.POOLING_LAYERS,
        use_upos: bool = defaults.USE_UPOS,
        use_xpos: bool = defaults.USE_XPOS,
        use_lemma: bool = defaults.USE_LEMMA,
        use_feats: bool = defaults.USE_FEATS,
        use_parse: bool = defaults.USE_PARSE,
        # Specific to the parser.
        arc_mlp_size: int = defaults.MLP_SIZE,
        label_mlp_size: int = defaults.MLP_SIZE,
        # Optimization.
        encoder_optimizer: cli.OptimizerCallable = defaults.OPTIMIZER,
        encoder_scheduler: cli.LRSchedulerCallable = defaults.SCHEDULER,
        classifier_optimizer: cli.OptimizerCallable = defaults.OPTIMIZER,
        classifier_scheduler: cli.LRSchedulerCallable = defaults.SCHEDULER,
        # Dummy values.
        upos_out_size: int = 2,  # Dummy value filled in via link.
        xpos_out_size: int = 2,  # Dummy value filled in via link.
        lemma_out_size: int = 2,  # Dummy value filled in via link.
        feats_out_size: int = 2,  # Dummy value filled in via link.
        label_out_size: int = 2,  # Dummy value filled in via link.
    ):
        super().__init__()
        # See what this disables here:
        # https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html#manual-optimization
        self.automatic_optimization = False
        self.encoder = modules.UDTubeEncoder(dropout, encoder, pooling_layers)
        self.classifier = modules.UDTubeClassifier(
            self.encoder.hidden_size,
            use_upos=use_upos,
            use_xpos=use_xpos,
            use_lemma=use_lemma,
            use_feats=use_feats,
            use_parse=use_parse,
            dropout=dropout,
            arc_mlp_size=arc_mlp_size,
            label_mlp_size=label_mlp_size,
            upos_out_size=upos_out_size,
            xpos_out_size=xpos_out_size,
            lemma_out_size=lemma_out_size,
            feats_out_size=feats_out_size,
            label_out_size=label_out_size,
        )
        self.loss_func = nn.CrossEntropyLoss(ignore_index=special.PAD_IDX)
        self.upos_accuracy = (
            self._make_accuracy(upos_out_size) if use_upos else None
        )
        self.xpos_accuracy = (
            self._make_accuracy(xpos_out_size) if use_xpos else None
        )
        self.lemma_accuracy = (
            self._make_accuracy(lemma_out_size) if use_lemma else None
        )
        self.feats_accuracy = (
            self._make_accuracy(feats_out_size) if use_feats else None
        )
        self.uas = (
            metrics.UnlabeledAttachmentScore(ignore_index=special.PAD_IDX)
            if use_parse
            else None
        )
        self.las = (
            metrics.LabeledAttachmentScore(ignore_index=special.PAD_IDX)
            if use_parse
            else None
        )
        self.encoder_optimizer = encoder_optimizer
        self.encoder_scheduler = encoder_scheduler
        self.classifier_optimizer = classifier_optimizer
        self.classifier_scheduler = classifier_scheduler
        self.save_hyperparameters()

    @staticmethod
    def _make_accuracy(num_classes: int) -> classification.MulticlassAccuracy:
        return classification.MulticlassAccuracy(
            num_classes, average="micro", ignore_index=special.PAD_IDX
        )

    @property
    def use_upos(self) -> bool:
        return self.classifier.use_upos

    @property
    def use_xpos(self) -> bool:
        return self.classifier.use_xpos

    @property
    def use_lemma(self) -> bool:
        return self.classifier.use_lemma

    @property
    def use_feats(self) -> bool:
        return self.classifier.use_feats

    @property
    def use_parse(self) -> bool:
        return self.classifier.use_parse

    def forward(
        self,
        batch: data.Batch,
    ) -> data.Logits:
        return self.classifier(self.encoder(batch))

    def configure_optimizers(
        self,
    ) -> List[Dict]:
        """Prepare optimizers and schedulers."""
        encoder_optimizer = self.encoder_optimizer(self.encoder.parameters())
        encoder_scheduler = self.encoder_scheduler(encoder_optimizer)
        encoder_dict = {
            "optimizer": encoder_optimizer,
            "lr_scheduler": {
                "scheduler": encoder_scheduler,
                "name": "encoder_lr",
            },
        }
        classifier_optimizer = self.classifier_optimizer(
            self.classifier.parameters()
        )
        classifier_scheduler = self.classifier_scheduler(classifier_optimizer)
        classifier_dict = {
            "optimizer": classifier_optimizer,
            "lr_scheduler": {
                "scheduler": classifier_scheduler,
                "name": "classifier_lr",
            },
        }
        return [encoder_dict, classifier_dict]

    # See the following for how these are called by the different subcommands.
    # https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#hooks

    def on_fit_start(self) -> None:
        # Rather than crashing, we simply warn about lack of deterministic
        # algorithms.
        if torch.are_deterministic_algorithms_enabled():
            torch.use_deterministic_algorithms(True, warn_only=True)

    def predict_step(self, batch: data.Batch, batch_idx: int) -> data.Logits:
        return self(batch)

    def training_step(
        self,
        batch: data.Batch,
        batch_idx: int,
    ) -> None:
        for optimizer in self.optimizers():
            optimizer.zero_grad()
        logits = self(batch)
        loss = self._log_loss(logits, batch, "train")
        self.manual_backward(loss)
        for optimizer in self.optimizers():
            optimizer.step()

    def on_train_epoch_end(self) -> None:
        for scheduler in self.lr_schedulers():
            # Users are advised to use the subclass defined in LightningCLI,
            # which has the name of the monitored variable encoded in it.
            if isinstance(scheduler, cli.ReduceLROnPlateau):
                scheduler.step(
                    metrics=self.trainer.callback_metrics[scheduler.monitor]
                )
            else:
                scheduler.step()

    def on_validation_epoch_start(self) -> None:
        self._reset_metrics()

    def validation_step(
        self,
        batch: data.Batch,
        batch_idx: int,
    ) -> None:
        logits = self(batch)
        self._log_loss(logits, batch, "val")
        self._update_metrics(logits, batch)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics_epoch_end("val")

    def on_test_step_epoch_start(self) -> None:
        self._reset_metrics()

    def test_step(self, batch: data.Batch, batch_idx: int) -> None:
        logits = self(batch)
        self._update_metrics(logits, batch)

    def on_test_epoch_end(self) -> None:
        self._log_metrics_epoch_end("test")

    def _log_loss(
        self, logits: data.Logits, batch: data.Batch, subset: str
    ) -> torch.Tensor:
        losses = []
        if self.use_upos:
            losses.append(self.loss_func(logits.upos, batch.upos))
        if self.use_xpos:
            losses.append(self.loss_func(logits.xpos, batch.xpos))
        if self.use_lemma:
            losses.append(self.loss_func(logits.lemma, batch.lemma))
        if self.use_feats:
            losses.append(self.loss_func(logits.feats, batch.feats))
        if self.use_parse:
            head_loss, label_loss = self.classifier.parser.compute_loss(
                logits.head,
                batch.head,
                logits.label,
                batch.label,
            )
            # TODO(kbg): maybe something more sophisticated or general is
            # required here; test later.
            # We weight the two losses as much as a single task.
            losses.append(head_loss / 2 + label_loss / 2)
        loss = torch.sum(torch.stack(losses))
        self.log(
            f"{subset}_loss",
            loss,
            batch_size=len(batch),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        # We can use the returned loss to step the optimizers.
        return loss

    def _reset_metrics(self) -> None:
        if self.use_upos:
            self.upos_accuracy.reset()
        if self.use_xpos:
            self.xpos_accuracy.reset()
        if self.use_lemma:
            self.lemma_accuracy.reset()
        if self.use_feats:
            self.feats_accuracy.reset()
        if self.use_parse:
            self.uas.reset()
            self.las.reset()

    def _update_metrics(self, logits: data.Logits, batch: data.Batch) -> None:
        if self.use_upos:
            self.upos_accuracy.update(logits.upos, batch.upos)
        if self.use_xpos:
            self.xpos_accuracy.update(logits.xpos, batch.xpos)
        if self.use_lemma:
            self.lemma_accuracy.update(logits.lemma, batch.lemma)
        if self.use_feats:
            self.feats_accuracy.update(logits.feats, batch.feats)
        if self.use_parse:
            self.uas.update(logits.head, batch.head)
            self.las.update(logits.head, batch.head, logits.label, batch.label)

    def _log_metrics_epoch_end(self, subset: str) -> None:
        if self.use_upos:
            self.log(
                f"{subset}_upos_accuracy",
                self.upos_accuracy.compute(),
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
        if self.use_xpos:
            self.log(
                f"{subset}_xpos_accuracy",
                self.xpos_accuracy.compute(),
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
        if self.use_lemma:
            self.log(
                f"{subset}_lemma_accuracy",
                self.lemma_accuracy.compute(),
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
        if self.use_feats:
            self.log(
                f"{subset}_feats_accuracy",
                self.feats_accuracy.compute(),
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
        if self.use_parse:
            self.log(
                f"{subset}_uas",
                self.uas.compute(),
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
            self.log(
                f"{subset}_las",
                self.las.compute(),
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
