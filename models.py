"""Models."""

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import joblib
import pytorch_lightning as pl
import torch
import transformers
from torch import nn, optim
from torchmetrics.functional import classification

import edit_scripts
import encoders
from batch import ConlluBatch, CustomBatch, TextBatch


class LabelZeroException(Exception):
    pass


class UDTube(pl.LightningModule):
    """The UDTube model.

    UDTube simultaneously handles POS tagging, lemmatization, and
    morphological feature tagging using a single BERT-style pre-trained,
    fine-tuned encoder.
    """

    def __init__(
        self,
        path_name: str = "UDTube",
        model_name: str = "bert-base-multilingual-cased",
        pos_out_label_size: int = 2,
        xpos_out_label_size: int = 2,
        lemma_out_label_size: int = 2,
        feats_out_label_size: int = 2,
        overall_dropout: float = 0.3,
        encoder_dropout: float = 0.5,
        warmup_steps: int = 500,
        overall_learning_rate: float = 5e-3,
        encoder_learning_rate: float = 2e-5,
        pooling_layers: int = 4,
        reverse_edits: bool = False,
        checkpoint: str = None,
        use_pos: bool = True,
        use_xpos: bool = True,
        use_lemma: bool = True,
        use_feats: bool = True,
    ):
        """Initializes the instance based on user input.

        Args:
            path_name: The name of the folder to save/load model assets. This includes the lightning logs and the encoders.
            model_name: The name of the model; used to tokenize and encode.
            pos_out_label_size: The amount of POS labels. This is usually passed by the dataset.
            xpos_out_label_size: The amount of language specific POS labels. This is usually passed by the dataset.
            lemma_out_label_size: The amount of lemma rule labels in the dataset. This is usually passed by the dataset.
            feats_out_label_size: The amount of feature labels in the dataset. This is usually passed by the dataset.
            overall_learning_rate: The learning rate of the full model
            encoder_learning_rate: The learning rate of only the encoder
            pooling_layers: The amount of layers used for embedding calculation
            checkpoint: The model checkpoint file
            use_pos: Whether the model will do POS tagging or not
            use_xpos: Whether not the model will do XPOS tagging or not
            use_lemma: Whether the model will do lemmatization or not
            use_feats: Whether the model will do Ufeats tagging or not
        """
        super().__init__()
        self._validate_input(
            pos_out_label_size, lemma_out_label_size, feats_out_label_size
        )
        self.path_name = path_name
        self.overall_learning_rate = overall_learning_rate
        self.encoder_learning_rate = encoder_learning_rate
        self.encoder_dropout = encoder_dropout
        self.warmup_steps = warmup_steps
        self.step_adjustment = None
        self.pooling_layers = pooling_layers
        self.encoder_layer = encoders.load(model_name, encoder_dropout)
        # Freezes encoder layer params for the first epoch.
        for p in self.encoder_layer.parameters():
            p.requires_grad = False
        logging.info("Encoder parameters frozen for the first epoch")
        self.dropout_layer = nn.Dropout(overall_dropout)
        # TODO make heads nullable
        if use_pos:
            self.pos_head = nn.Sequential(
                nn.Linear(
                    self.encoder_layer.config.hidden_size, pos_out_label_size
                ),
                nn.LeakyReLU(),
            )
        if use_xpos:
            self.xpos_head = nn.Sequential(
                nn.Linear(
                    self.encoder_layer.config.hidden_size, xpos_out_label_size
                ),
                nn.LeakyReLU(),
            )
        if use_lemma:
            self.lemma_head = nn.Sequential(
                nn.Linear(
                    self.encoder_layer.config.hidden_size, lemma_out_label_size
                ),
                nn.LeakyReLU(),
            )
        if use_feats:
            self.feats_head = nn.Sequential(
                nn.Linear(
                    self.encoder_layer.config.hidden_size, feats_out_label_size
                ),
                nn.LeakyReLU(),
            )
        self.use_pos = use_pos
        self.use_xpos = use_xpos
        self.use_lemma = use_lemma
        self.use_feats = use_feats
        # retrieving the LabelEncoders set up by the Dataset
        self.lemma_encoder = joblib.load(
            f"{self.path_name}/lemma_encoder.joblib"
        )
        self.feats_encoder = joblib.load(
            f"{self.path_name}/ufeats_encoder.joblib"
        )
        self.pos_encoder = joblib.load(f"{self.path_name}/upos_encoder.joblib")
        self.xpos_encoder = joblib.load(
            f"{self.path_name}/xpos_encoder.joblib"
        )
        self.e_script = (
            edit_scripts.ReverseEditScript
            if reverse_edits
            else edit_scripts.EditScript
        )
        self.save_hyperparameters()
        self.dummy_tensor = torch.zeros(
            self.encoder_layer.config.hidden_size, device=self.device
        )
        if checkpoint:
            checkpoint = torch.load(checkpoint)
            logging.info("Loading from checkpoint")
            self.load_state_dict(checkpoint["state_dict"])

    def _validate_input(
        self, pos_out_label_size, lemma_out_label_size, feats_out_label_size
    ):
        if (
            pos_out_label_size == 0
            or lemma_out_label_size == 0
            or feats_out_label_size == 0
        ):
            raise LabelZeroException(
                "One of the label sizes given to the model was zero"
            )

    def pad_seq(
        self,
        sequence: Iterable,
        pad: Iterable,
        max_len: int,
        return_long: bool = False,
    ) -> torch.Tensor:
        # TODO find a way to keep this on GPU
        padded_seq = []
        for s in sequence:
            if not torch.is_tensor(s):
                s = torch.tensor(s, device=self.device)
            if len(s) != max_len:
                # I think the below operation puts the padding back on CPU, not great
                r_padding = torch.stack([pad] * (max_len - len(s)))
                r_padding = r_padding.to(s.device)
                padded_seq.append(torch.cat((s, r_padding)))
            else:
                padded_seq.append(s)
        if return_long:
            return torch.stack(padded_seq).long()
        return torch.stack(padded_seq)

    def pool_embeddings(
        self,
        x_embs: torch.Tensor,
        tokenized: transformers.BatchEncoding,
        gold_label_ex=None,
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor, int]:
        new_embs = []
        new_masks = []
        words = []
        encodings = tokenized.encodings
        offset = 1
        if not encodings:
            encodings = tokenized.custom_encodings
            offset = 0  # no offset here, this is the Byt5 case
        for encoding, x_emb_i in zip(encodings, x_embs):
            embs_i = []
            words_i = []
            mask_i = []
            last_word_idx = slice(0, 0)
            # skipping over the first padding token ([CLS])
            for word_id, x_emb_j in zip(
                encoding.word_ids[offset:], x_emb_i[offset:]
            ):
                if word_id is None:
                    # emb padding
                    break
                start, end = encoding.word_to_tokens(word_id)
                word_idxs = slice(start, end)
                if word_idxs != last_word_idx:
                    last_word_idx = word_idxs
                    word_emb_pooled = torch.mean(
                        x_emb_i[word_idxs], keepdim=True, dim=0
                    ).squeeze()
                    embs_i.append(word_emb_pooled)
                    try:
                        words_i.append(
                            "".join(encoding.tokens[word_idxs])
                            .replace("##", "")
                            .replace("▁", "")
                        )
                    except TypeError:
                        words_i.append(
                            bytes(encoding.tokens[word_idxs]).decode()
                        )
                    mask_i.append(1)
            words.append(words_i)
            new_embs.append(torch.stack(embs_i))
            new_masks.append(torch.tensor(mask_i, device=self.device))
        if gold_label_ex:
            longest_seq = max(
                max(len(m) for m in words), max(len(l) for l in gold_label_ex)
            )
        else:
            longest_seq = max(len(m) for m in words)
        new_embs = self.pad_seq(new_embs, self.dummy_tensor, longest_seq)
        new_masks = self.pad_seq(
            new_masks, torch.tensor(0, device=self.device), longest_seq
        )
        return new_embs, words, new_masks, longest_seq

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        grouped_params = [
            {
                "params": self.encoder_layer.parameters(),
                "lr": self.encoder_learning_rate,
                "weight_decay": 0.01,
            }
        ]
        if self.use_lemma:
            grouped_params.append(
                {
                    "params": self.lemma_head.parameters(),
                    "lr": self.overall_learning_rate,
                }
            )
        if self.use_pos:
            grouped_params.append(
                {
                    "params": self.pos_head.parameters(),
                    "lr": self.overall_learning_rate,
                }
            )
        if self.use_xpos:
            grouped_params.append(
                {
                    "params": self.xpos_head.parameters(),
                    "lr": self.overall_learning_rate,
                }
            )
        if self.use_feats:
            grouped_params.append(
                {
                    "params": self.feats_head.parameters(),
                    "lr": self.overall_learning_rate,
                }
            )
        optimizer = torch.optim.AdamW(grouped_params)
        return [optimizer]

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[
            optim.Optimizer, pl.core.optimizer.LightningOptimizer
        ],
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        optimizer.step(closure=optimizer_closure)
        if epoch > 0:
            # this only happens once params are unfrozen
            unfrozen_steps = self.trainer.global_step - self.step_adjustment
            if unfrozen_steps < self.warmup_steps:
                # warming up LR from 0 to LR (ENCODER ONLY), starts before the encoder is unfrozen so LR is not at 0 during meaningful steps
                lr_scale = 0 + float(unfrozen_steps) / self.warmup_steps
            else:
                # decaying weight
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
                batch_size=32,  # TODO pass this instead
            )

    def log_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        batch_size: int,
        task_name: str,
        subset: str = "train",
        ignore_index: int = 0,
    ) -> None:
        num_classes = y_pred.shape[1]
        # getting the pad
        acc = classification.multiclass_accuracy(
            y_pred.permute(2, 1, 0),
            y_true.T,
            num_classes=num_classes,
            ignore_index=ignore_index,
            average="micro",
        )
        self.log(
            f"{subset}_{task_name}_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

    def _call_head(
        self,
        y_gold,
        logits,
        task_name="pos",
        subset="train",
        return_loss=True,
    ):
        batch_size, longest_seq, num_classes = logits.shape
        # ignore_idx is used for padding!
        lencoder = getattr(self, f"{task_name}_encoder")
        ignore_idx = int(lencoder.transform(["[PAD]"])[0])  # TODO better way?
        pad = torch.tensor(ignore_idx, device=self.device)
        y_gold_tensor = self.pad_seq(
            y_gold, pad, longest_seq, return_long=True
        )

        # Each head returns ( batch X sequence_len X classes )
        # but CE & metrics want ( minibatch X Classes X sequence_len ); (minibatch, C, d0...dk) & (N, C, ..) in the docs
        logits = logits.permute(0, 2, 1)
        self.log_metrics(
            logits,
            y_gold_tensor,
            batch_size,
            task_name,
            subset=subset,
            ignore_index=ignore_idx,
        )
        if return_loss:
            loss = nn.functional.cross_entropy(
                logits, y_gold_tensor, ignore_index=ignore_idx
            )
            return loss

    def _decode_to_str(
        self,
        sentences,
        words,
        y_pos_logits,
        y_xpos_logits,
        y_lemma_logits,
        y_feats_logits,
        replacements=None,
    ) -> Tuple[str]:
        # argmaxing
        if self.use_pos:
            y_pos_hat = torch.argmax(y_pos_logits, dim=-1)
        if self.use_xpos:
            y_xpos_hat = torch.argmax(y_xpos_logits, dim=-1)
        if self.use_lemma:
            y_lemma_hat = torch.argmax(y_lemma_logits, dim=-1)
        if self.use_feats:
            y_feats_hat = torch.argmax(y_feats_logits, dim=-1)
        # transforming to str
        (
            y_pos_str_batch,
            y_xpos_str_batch,
            y_lemma_str_batch,
            y_feats_hat_batch,
        ) = ([], [], [], [])
        for batch_idx, w_i in enumerate(words):
            lemmas_i = []
            seq_len = len(w_i)  # used to get rid of padding
            if self.use_pos:
                y_pos_str_batch.append(
                    self.pos_encoder.inverse_transform(
                        y_pos_hat[batch_idx][:seq_len].cpu()
                    )
                )
            if self.use_xpos:
                y_xpos_str_batch.append(
                    self.xpos_encoder.inverse_transform(
                        y_xpos_hat[batch_idx][:seq_len].cpu()
                    )
                )
            if self.use_feats:
                y_feats_hat_batch.append(
                    self.feats_encoder.inverse_transform(
                        y_feats_hat[batch_idx][:seq_len].cpu()
                    )
                )

            if self.use_lemma:
                # lemmas work through rule classification, so we have to also apply the rules.
                lemma_rules = self.lemma_encoder.inverse_transform(
                    y_lemma_hat[batch_idx][:seq_len].cpu()
                )
                for i, rule in enumerate(lemma_rules):
                    rscript = self.e_script.fromtag(rule)
                    lemma = rscript.apply(words[batch_idx][i])
                    lemmas_i.append(lemma)
                y_lemma_str_batch.append(lemmas_i)

        return (
            sentences,
            words,
            y_pos_str_batch,
            y_xpos_str_batch,
            y_lemma_str_batch,
            y_feats_hat_batch,
            replacements,
        )

    def forward(
        self, batch: CustomBatch
    ) -> Tuple[Iterable[str], list[list[str]], torch.Tensor]:
        # if something is longer than an allowed sequence, we have to trim it down
        if (
            batch.tokens.input_ids.shape[1]
            >= self.encoder_layer.config.max_position_embeddings
        ):
            logging.warning(
                f"trimmed sequence down to maximum seq_len allowed: {self.encoder_layer.config.max_position_embeddings}"
            )
            batch.tokens.input_ids = batch.tokens.input_ids[
                : self.encoder_layer.config.max_position_embeddings
            ]
            batch.tokens.attention_mask = batch.tokens.attention_mask[
                : self.encoder_layer.config.max_position_embeddings
            ]
        # getting raw embeddings
        x = self.encoder_layer(
            batch.tokens.input_ids.to(self.device),
            batch.tokens.attention_mask.to(self.device),
        )
        # stacking n (self.pooling_layer) embedding layers
        x = torch.stack(x.hidden_states[-self.pooling_layers :])
        # averages n layers into one embedding layer
        x = torch.mean(x, keepdim=True, dim=0).squeeze()
        if len(x.shape) == 2:
            # batch of size one
            x = x.unsqueeze(dim=0)
        x = self.dropout_layer(x)
        # this is used for padding, when present
        if isinstance(batch, TextBatch):
            gold_label_ex = None
        else:
            gold_label_ex = batch.pos
        # converting x embeddings to the word level
        x, words, masks, longest_seq = self.pool_embeddings(
            x, batch.tokens, gold_label_ex=gold_label_ex
        )
        y_pos_logits = self.pos_head(x) if self.use_pos else None
        y_xpos_logits = self.xpos_head(x) if self.use_xpos else None
        y_lemma_logits = self.lemma_head(x) if self.use_lemma else None
        y_feats_logits = self.feats_head(x) if self.use_feats else None
        # TODO make the model response an object (and change type hints accordingly)
        return (
            batch.sentences,
            words,
            y_pos_logits,
            y_xpos_logits,
            y_lemma_logits,
            y_feats_logits,
            masks,
        )

    def training_step(
        self, batch: ConlluBatch, batch_idx: int, subset: str = "train"
    ) -> Dict[str, float]:
        response = {}
        batch_size = len(batch)
        (
            sentences,
            words,
            y_pos_logits,
            y_xpos_logits,
            y_lemma_logits,
            y_feats_logits,
            masks,
        ) = self(batch)
        if self.use_pos:
            pos_loss = self._call_head(
                batch.pos,
                y_pos_logits,
                task_name="pos",
                subset=subset,
            )
            response["pos_loss"] = pos_loss
        if self.use_xpos:
            xpos_loss = self._call_head(
                batch.xpos, y_xpos_logits, task_name="xpos", subset=subset
            )
            response["xpos_loss"] = xpos_loss
        if self.use_lemma:
            lemma_loss = self._call_head(
                batch.lemmas,
                y_lemma_logits,
                task_name="lemma",
                subset=subset,
            )
            response["lemma_loss"] = lemma_loss
        if self.use_feats:
            feats_loss = self._call_head(
                batch.feats,
                y_feats_logits,
                task_name="feats",
                subset=subset,
            )
            response["feats_loss"] = feats_loss
        # combining the loss of the active heads
        loss = torch.mean(torch.stack([l for l in response.values()]))
        response["loss"] = loss
        self.log(
            f"{subset}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        return response

    def on_train_epoch_end(self) -> None:
        if self.current_epoch == 0:
            # how many steps are in an epoch
            self.step_adjustment = self.trainer.global_step
            for p in self.encoder_model.parameters():
                p.requires_grad = True
            logging.info("Encoder parameters unfrozen")

    def validation_step(
        self, batch: ConlluBatch, batch_idx: int
    ) -> Dict[str, float]:
        return self.training_step(batch, batch_idx, subset="val")

    def test_step(self, batch: ConlluBatch, batch_idx: int) -> Tuple[str]:
        (
            sentences,
            words,
            y_pos_logits,
            y_xpos_logits,
            y_lemma_logits,
            y_feats_logits,
            masks,
        ) = self(batch)
        return self._decode_to_str(
            sentences,
            words,
            y_pos_logits,
            y_xpos_logits,
            y_lemma_logits,
            y_feats_logits,
            replacements=batch.replacements,
        )

    def predict_step(self, batch: TextBatch, batch_idx: int) -> Tuple[str]:
        *res, _ = self(batch)
        return self._decode_to_str(*res, replacements=batch.replacements)
