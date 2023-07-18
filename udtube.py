from typing import Iterable

import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
import torch
import transformers
from torch import nn, tensor
from torchmetrics import Accuracy, F1Score, Precision, Recall

from batch import InferenceBatch, TrainBatch
from conllu_datasets import UPOS_CLASSES
from data_module import ConlluDataModule


class UDTubeCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.pos_classes_cnt", "model.pos_out_label_size", apply_on="instantiate")
        parser.link_arguments("data.lemma_classes_cnt", "model.lemma_out_label_size", apply_on="instantiate")
        parser.link_arguments("data.feats_classes_cnt", "model.feats_out_label_size", apply_on="instantiate")


class UDTube(pl.LightningModule):
    """The main model file

    UDTube is a BERT based model that handles 3 tasks at once, pos tagging, lemmatization, and feature classification."""
    def __init__(
            self,
            model_name: str = "bert-base-multilingual-cased",
            pos_out_label_size: int = 2,
            lemma_out_label_size: int = 2,
            feats_out_label_size: int = 2,
            learning_rate: float = 0.001,
            pooling_layers: int = 4
    ):
        """Initializes the instance based on spam preference.

        Args:
          model_name: The name of the model; used to tokenize and encode.
          pos_out_label_size: The amount of POS labels. This is usually passed by the dataset.
          lemma_out_label_size: The amount of lemma rule labels in the dataset. This is usually passed by the dataset.
          feats_out_label_size: The amount of feature labels in the dataset. This is usually passed by the dataset.
          learning_rate: The learning rate
          pooling_layers: The amount of layers used for embedding calculation
        """
        super().__init__()
        self.bert = transformers.AutoModel.from_pretrained(
            model_name, output_hidden_states=True
        )
        self.learning_rate = learning_rate
        self.pooling_layers = pooling_layers
        self.pos_pad = tensor(
            pos_out_label_size - 1
        )  # last item in the list is a pad from the label encoder
        self.lemma_pad = tensor(lemma_out_label_size)
        self.feats_pad = tensor(feats_out_label_size)
        self.pos_head = nn.Sequential(
            nn.Linear(
                self.bert.config.hidden_size, pos_out_label_size
            ),
            nn.Tanh()
        )
        self.lemma_head = nn.Sequential(
            nn.Linear(
                self.bert.config.hidden_size, lemma_out_label_size + 1
            ),  # + 1 for padding labels (for now)
            nn.Tanh(),
        )
        self.feats_head = nn.Sequential(
            nn.Linear(
                self.bert.config.hidden_size, feats_out_label_size + 1
            ),  # + 1 for padding labels (for now)
            nn.Tanh(),
        )

        # Setting up all the metrics objects for each task
        self.pos_loss = nn.CrossEntropyLoss(ignore_index=self.pos_pad.item())
        self.pos_accuracy = Accuracy(
            task="multiclass",
            num_classes=pos_out_label_size,
            ignore_index=self.pos_pad.item(),
        )

        self.lemma_loss = nn.CrossEntropyLoss(
            ignore_index=self.lemma_pad.item()
        )
        self.lemma_accuracy = Accuracy(
            task="multiclass",
            num_classes=lemma_out_label_size + 1,
            ignore_index=self.lemma_pad.item(),
        )

        self.feats_loss = nn.CrossEntropyLoss(
            ignore_index=self.feats_pad.item()
        )
        self.feats_accuracy = Accuracy(
            task="multiclass",
            num_classes=feats_out_label_size + 1,
            ignore_index=self.feats_pad.item(),
        )

    def pad_seq(
            self,
            sequence: Iterable,
            pad: Iterable,
            max_len: int,
            return_long: bool = False,
    ):
        padded_seq = []
        for s in sequence:
            if len(s) != max_len:
                r_padding = torch.stack([pad] * (max_len - len(s)))
                padded_seq.append(torch.cat((s, r_padding)))
            else:
                padded_seq.append(s)
        if return_long:
            return torch.stack(padded_seq).long()
        return torch.stack(padded_seq)

    def pool_embeddings(
            self, x_embs: torch.tensor, tokenized: transformers.BatchEncoding
    ):
        new_embs = []
        new_masks = []
        for encoding, x_emb_i in zip(tokenized.encodings, x_embs):
            embs_i = []
            mask_i = []
            last_word_idx = slice(0, 0)
            for word_id, x_emb_j in zip(encoding.word_ids, x_emb_i):
                if word_id is None:
                    embs_i.append(x_emb_j)
                    # TODO maybe make dummy tensor a member of class
                    dummy_tensor = x_emb_j
                    mask_i.append(0)
                    continue
                start, end = encoding.word_to_tokens(word_id)
                word_idxs = slice(start, end)
                if word_idxs != last_word_idx:
                    last_word_idx = word_idxs
                    word_emb_pooled = torch.mean(
                        x_emb_i[word_idxs], keepdim=True, dim=0
                    ).squeeze()
                    embs_i.append(word_emb_pooled)
                    mask_i.append(1)
            new_embs.append(torch.stack(embs_i))
            new_masks.append(tensor(mask_i))
        longest_seq = max(len(m) for m in new_masks)
        new_embs = self.pad_seq(new_embs, dummy_tensor, longest_seq)
        new_masks = self.pad_seq(new_masks, tensor(0), longest_seq)
        return new_embs, new_masks, longest_seq

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def log_metrics(
            self,
            y_pred: torch.tensor,
            y_true: torch.tensor,
            batch_size: int,
            task_name: str,
            subset: str = "train",
    ):
        accuracy = getattr(self, f"{task_name}_accuracy")
        self.log(
            f"{subset}:{task_name}_acc",
            accuracy(y_pred, y_true),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size
        )

    def forward(self, batch: InferenceBatch):
        x_encoded = self.bert(
            batch.tokens.input_ids, batch.tokens.attention_mask
        )
        last_n_layer_embs = torch.stack(x_encoded.hidden_states[-self.pooling_layers:])
        x_embs = torch.mean(last_n_layer_embs, keepdim=True, dim=0).squeeze()
        x_word_embs, attn_masks, longest_seq = self.pool_embeddings(
            x_embs, batch.tokens
        )

        y_pos_logits = self.pos_head(x_word_embs)
        y_lemma_logits = self.lemma_head(x_word_embs)
        y_feats_logits = self.feats_head(x_word_embs)

        return y_pos_logits, y_lemma_logits, y_feats_logits

    def training_step(self, batch: TrainBatch, batch_idx: int, subset: str = "train"):
        x_encoded = self.bert(
            batch.tokens.input_ids, batch.tokens.attention_mask
        )
        last_4_layer_embs = torch.stack(x_encoded.hidden_states[-4:])
        x_embs = torch.mean(last_4_layer_embs, keepdim=True, dim=0).squeeze()
        x_word_embs, attn_masks, longest_seq = self.pool_embeddings(
            x_embs, batch.tokens
        )

        # need to do some preprocessing on Y
        y_pos_tensor = self.pad_seq(
            batch.pos, self.pos_pad, longest_seq, return_long=True
        )  # TODO passing self. is weird
        y_lemma_tensor = self.pad_seq(
            batch.lemmas, self.lemma_pad, longest_seq, return_long=True
        )
        y_feats_tensor = self.pad_seq(
            batch.feats, self.feats_pad, longest_seq, return_long=True
        )

        # getting logits from each head, and then permuting them for metrics calculation
        # Each head returns batch X sequence_len X classes
        # but CE and metrics want minibatch X Classes X sequence_len, (minibatch, C, d0...dk) & (N, C, ..) in the docs.
        y_pos_logits = self.pos_head(x_word_embs)
        y_pos_logits = y_pos_logits.permute(0, 2, 1)

        y_lemma_logits = self.lemma_head(x_word_embs)
        y_lemma_logits = y_lemma_logits.permute(0, 2, 1)

        y_feats_logits = self.feats_head(x_word_embs)
        y_feats_logits = y_feats_logits.permute(0, 2, 1)

        # getting loss and logging for each head
        batch_size = len(batch)

        pos_loss = self.pos_loss(y_pos_logits, y_pos_tensor)
        self.log_metrics(y_pos_logits, y_pos_tensor, batch_size, "pos", subset=subset)

        lemma_loss = self.lemma_loss(y_lemma_logits, y_lemma_tensor)
        self.log_metrics(
            y_lemma_logits, y_lemma_tensor, batch_size, "lemma", subset=subset
        )

        feats_loss = self.feats_loss(y_feats_logits, y_feats_tensor)
        self.log_metrics(
            y_feats_logits, y_feats_tensor, batch_size, "feats", subset=subset
        )

        # combining the loss of the heads
        loss = torch.mean(torch.stack([pos_loss, lemma_loss, feats_loss]))
        self.log(
            "Loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        return self.training_step(batch, batch_idx, subset="val")


if __name__ == "__main__":
    cli = UDTubeCLI(UDTube, ConlluDataModule)


