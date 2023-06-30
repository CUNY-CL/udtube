import torch
import pytorch_lightning as pl
import transformers
import logging

from torch import nn
from torchmetrics import Accuracy, F1Score, Recall, Precision
from torch.utils.data import random_split, DataLoader
from conllu_datasets import ConlluMapDataset, UPOS_CLASSES

from batch import TrainBatch, InferenceBatch

LOGGER = logging.getLogger('UDtube')
BERT_MAX_LEN = 512


class UDTube(pl.LightningModule):
    def __init__(self, model_name: str, pos_out_label_size: int = 2, lemma_out_label_size: int = 2, ufeats_out_label_size: int = 2):
        super().__init__()
        self.bert = transformers.AutoModel.from_pretrained(model_name)
        self.pos_pad = pos_out_label_size - 1  # last item in the list is a pad from the label encoder
        self.lemma_pad = lemma_out_label_size
        self.ufeats_pad = ufeats_out_label_size
        self.pos_head = nn.Sequential(
            nn.Linear(768, pos_out_label_size),
            nn.Tanh()
        )
        self.lemma_head = nn.Sequential(
            nn.Linear(768, lemma_out_label_size + 1),  # + 1 for padding labels (for now)
            nn.Tanh()
        )
        self.ufeats_head = nn.Sequential(
            nn.Linear(768, ufeats_out_label_size + 1),  # + 1 for padding labels (for now)
            nn.Tanh()
        )
        self.loss = nn.CrossEntropyLoss()

    def pool_embeddings(self, x_embs, id_to_word_mappings):
        """Pool token embeddings together by averaging at the word level"""
        # TODO what tensor operations can help here?
        final_embs = []
        for i, batch in enumerate(x_embs):
            dummy_emb = torch.zeros(768)  # this is the 'padding' embedding
            new_embs = [dummy_emb] * BERT_MAX_LEN  # initializing a container full of dummy embeddings
            last_word_idx = -1
            pooler = []
            for j, word_idx in enumerate(id_to_word_mappings[i]):
                if j == 0:
                    # special case, [CLS] token
                    new_embs[j] = batch[j]
                    word_idx = 0
                elif word_idx is None:
                    pooled_emb = torch.mean(torch.stack(pooler), keepdim=True, dim=0).squeeze()
                    new_embs[last_word_idx] = pooled_emb
                    # want to break here, we are at padding
                    break
                elif last_word_idx == word_idx:
                    pooler.append(batch[j])
                else:
                    pooled_emb = torch.mean(torch.stack(pooler), keepdim=True, dim=0).squeeze()
                    new_embs[last_word_idx] = pooled_emb
                    # resetting the pooler to next word item
                    pooler = [batch[j]]
                last_word_idx = word_idx
            new_embs = torch.stack(new_embs)
            final_embs.append(new_embs)
        final_embs_tensor = torch.stack(final_embs)
        return final_embs_tensor

    def preprocess_target(self, y, pad):
        """Pad the target rows to have all the same length and return a y tensor,
        and a y list containing lists of indices """
        new_targ = []
        padding_indices = []
        for y_i in y:
            # remove padding step
            r_padding = [pad] * (BERT_MAX_LEN - len(y_i) - 1)
            r_padding_idxs = list(range(len([pad] + y_i), len(r_padding)))
            # note that LongTensors are required for CE loss
            y_i_tensor = torch.LongTensor([pad] + y_i + r_padding)  # the left pad is for [CLS]
            new_targ.append(y_i_tensor)
            padding_indices.append(
                [0] + r_padding_idxs  # 0 is constant; CLS is considered padding
            )
        return torch.stack(new_targ), padding_indices

    def remove_target_padding(self, y_pred, y_true, padding_indices):
        """CURRENTLY NOT USED Remove the padding from the target tensor and return a lis tof unpadded targets. """
        padless_y_pred = []
        padless_y_true = []
        for y_pred_row, y_true_row, padding_indices_row in zip(y_pred, y_true, padding_indices):
            padless_y_pred_row = []
            padless_y_true_row = []
            for i, (y_pred_i, y_true_i) in enumerate(zip(y_pred_row, y_true_row)):
                if i not in padding_indices_row:
                    padless_y_pred_row.append(y_pred_i)
                    padless_y_true_row.append(y_true_i)
            padless_y_pred.append(torch.stack(padless_y_pred_row))
            padless_y_true.append(torch.stack(padless_y_true_row))
        padless_y_pred = padless_y_pred
        padless_y_true = padless_y_true
        return padless_y_pred, padless_y_true

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def log_metrics(self, y_pred, y_true, padding_idxs, task_name: str, subset: str = "train"):
        task_to_out_label_size = {
            # todo something better than the pad
            "pos": self.pos_pad + 1,
            "lemma": self.lemma_pad + 1,
            "ufeats": self.ufeats_pad + 1
        }
        # is it multilabel?
        precision = Precision(task="multiclass", num_classes=task_to_out_label_size[task_name])
        recall = Recall(task="multiclass", num_classes=task_to_out_label_size[task_name])
        f1 = F1Score(task="multiclass", num_classes=task_to_out_label_size[task_name])
        accuracy = Accuracy(task="multiclass", num_classes=task_to_out_label_size[task_name])

        avg_precision = precision(y_pred, y_true)
        avg_recall = recall(y_pred, y_true)
        avg_f1s = f1(y_pred, y_true)
        avg_accs = accuracy(y_pred, y_true)

        self.log(f"{subset}:{task_name}_precision", avg_precision, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log(f"{subset}:{task_name}_recall", avg_recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{subset}:{task_name}_f1", avg_f1s, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{subset}:{task_name}_acc", avg_accs, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def forward(self, batch):
        # NOTE: Forward is completely untested at the moment
        # assuming testDataloader usage with preprocessing (need to find out otherwise)
        batch = InferenceBatch(*batch)

        x_encoded = self.bert(batch.ids)
        x_embs = x_encoded.last_hidden_state
        x_word_embs = self.pool_embeddings(x_embs, batch.word_mappings)

        y_pos_logits = self.pos_head(x_word_embs)
        y_lemma_logits = self.lemma_head(x_word_embs)
        y_ufeats_logits = self.ufeats_head(x_word_embs)

        y_pos_hat = torch.argmax(y_pos_logits, dim=-1)
        y_lemma_hat = torch.argmax(y_lemma_logits, dim=-1)
        y_ufeats_hat = torch.argmax(y_ufeats_logits, dim=-1)
        return y_pos_hat, y_lemma_hat, y_ufeats_hat

    def training_step(self, batch, batch_idx, subset: str = "train"):
        batch = TrainBatch(*batch)

        x_encoded = self.bert(batch.ids, batch.attention_masks)
        x_embs = x_encoded.last_hidden_state
        x_word_embs = self.pool_embeddings(x_embs, batch.word_mappings)

        # # need to do some preprocessing on Y
        y_pos_tensor, y_pos_padding_idxs = self.preprocess_target(batch.pos, self.pos_pad)  # TODO passing self. is weird
        y_lemma_tensor, y_lemma_padding_idxs = self.preprocess_target(batch.lemma, self.lemma_pad)
        y_ufeats_tensor, y_ufeats_padding_idxs = self.preprocess_target(batch.ufeats, self.ufeats_pad)

        # getting logits from each head, and then premuting them for metrics calculation
        y_pos_logits = self.pos_head(x_word_embs)
        y_pos_logits = y_pos_logits.permute(0, 2, 1)

        y_lemma_logits = self.lemma_head(x_word_embs)
        y_lemma_logits = y_lemma_logits.permute(0, 2, 1)

        y_ufeats_logits = self.ufeats_head(x_word_embs)
        y_ufeats_logits = y_ufeats_logits.permute(0, 2, 1)

        # getting loss and logging for each head
        pos_loss = self.loss(y_pos_logits, y_pos_tensor)
        self.log_metrics(y_pos_logits, y_pos_tensor, y_pos_padding_idxs, 'pos', subset=subset)

        lemma_loss = self.loss(y_lemma_logits, y_lemma_tensor)
        self.log_metrics(y_lemma_logits, y_lemma_tensor, y_lemma_padding_idxs, 'lemma', subset=subset)

        ufeats_loss = self.loss(y_ufeats_logits, y_ufeats_tensor)
        self.log_metrics(y_ufeats_logits, y_ufeats_tensor, y_ufeats_padding_idxs, 'ufeats', subset=subset)

        # combining the loss of the heads
        loss = (pos_loss + lemma_loss + ufeats_loss) / 3

        res = {
            "loss": loss
        }

        return res

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx, subset="val")
        return results


if __name__ == '__main__':
    # TODO get all this from parser
    bert_model = 'bert-base-multilingual-cased'
    dataset_path = 'el_gdt-ud-dev.conllu'
    num_epochs = 3
    batch_size = 8 # keeping it small for testing
    random_seed = 42

    # Loading in the data
    data = ConlluMapDataset(dataset_path)

    # making a validation set
    seed_for_reproducibility = torch.Generator().manual_seed(random_seed)
    train_data, val_data = random_split(data, [0.8, 0.2], seed_for_reproducibility)

    # initializing a tokenizer for preprocessing
    tokenizer = transformers.AutoTokenizer.from_pretrained(bert_model)

    # Loading the data into a dataloader with a preprocessing function to make X a tensor
    def preprocessing_func(batch):
        """Data pipeline -> tokenizing input and grouping token IDs to words. The output has varied dimensions"""
        sentences, *args = zip(*batch)  # make batches a class?
        X_ids = []
        X_word_mappings = []
        X_attention_masks = []

        tokenized_X = tokenizer(sentences, padding='max_length')
        for i, tokens in enumerate(tokenized_X.encodings):
            X_ids.append(torch.IntTensor(tokens.ids))
            X_word_mappings.append(tokens.words)
            X_attention_masks.append(torch.IntTensor(tokens.attention_mask))
        X_ids_tensor = torch.stack(X_ids)
        X_att_tensor = torch.stack(X_attention_masks)
        return X_ids_tensor, X_word_mappings, X_att_tensor, sentences, *args


    train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=preprocessing_func)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, collate_fn=preprocessing_func)

    # initializing the model
    pos_out_label_size = len(UPOS_CLASSES)
    ufeats_out_label_size = len(data.feats_classes)
    lemma_out_label_size = len(data.lemma_classes)
    LOGGER.info("Loading in Bert model, this may take a while")
    model = UDTube(bert_model, pos_out_label_size=pos_out_label_size,
                   lemma_out_label_size=lemma_out_label_size, ufeats_out_label_size=ufeats_out_label_size)

    # Training the model
    trainer = pl.Trainer(max_epochs=num_epochs)
    trainer.fit(model, train_dataloader, val_dataloader)
