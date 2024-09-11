import json
import re
from functools import partial
import os
import torch
import spacy_udpipe

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5TokenizerFast
from tokenizers import Encoding

from batch import ConlluBatch, TextBatch
from conllu_datasets import (
    ConlluMapDataset,
    TextIterDataset,
    ConlluIterDataset,
)
from typing import Optional

from defaults import SPECIAL_TOKENS

ROOT_TOKEN = "ROOT"


class CustomEncoding:
    # TODO will it break if we inherit from tokenizers.Encoding?
    """Encoding object that partially emulates tokenizer package encodings object.

    For models that are not built on rust(like Byt5) this object is needed to not break later code. In particular,
    the implementation of word_to_tokens.
    """

    def __init__(self, word_ids, tokens):
        self.word_ids = word_ids
        self.tokens = tokens

    def word_to_tokens(self, word_id):
        # TODO: less naive approach
        start = self.word_ids.index(word_id)
        stop = start + 1
        for i, idx in enumerate(self.word_ids[start + 1 :]):
            if idx == word_id:
                stop += 1
            else:
                break
        return start, stop


class ConlluDataModule(pl.LightningDataModule):
    """Reusable DataModule that manages stages of UDTube

    This class is initialized by the LightningCLI interface. It manages all the data loading steps for UDTube. Train/Val
    are loaded into memory, but prediction data is loaded in slowly to avoid memory errors when loading in large files.
    """

    def __init__(
        self,
        path_name: str = "UDTube",
        model_name: str = "bert-base-multilingual-cased",
        language: str = "en",
        train_dataset: Optional[str] = None,
        val_dataset: Optional[str] = None,
        predict_dataset: Optional[str] = None,
        test_dataset: Optional[str] = None,
        batch_size: int = 32,
        reverse_edits: bool = False,
        lang_with_space: bool = True,
        checkpoint: Optional[str] = None,
    ):
        """Initializes the instance based on user input. Some attributes will not be set if train dataset is None.

        Args:
            path_name: The name of the folder to save model assets in
            model_name: The name of the model; used to tokenize and encode.
            train_dataset: The path to the training conllu file
            val_dataset: The path to the validation conllu file
            predict_dataset: The path to the prediction dataset, can either be text or conllu
            batch_size: The batch_size
            reverse_edits: Reverse edit script calculation. Recommended for suffixal languages. False by default
            checkpoint: The path to the model checkpoint file
        """
        super().__init__()
        try:
            self.pretokenizer = spacy_udpipe.load(language)
        except:
            spacy_udpipe.download(language)
            self.pretokenizer = spacy_udpipe.load(language)
        self.reverse_edits = reverse_edits
        self.lang_with_space = lang_with_space
        self.train_dataset = ConlluMapDataset(
            train_dataset,
            reverse_edits=self.reverse_edits,
            path_name=path_name,
            train=True,
        )
        self.val_dataset = ConlluMapDataset(
            val_dataset,
            reverse_edits=self.reverse_edits,
            path_name=path_name,
            train=False,
        )
        if predict_dataset and predict_dataset.endswith(".conllu"):
            self.predict_dataset = ConlluIterDataset(predict_dataset)
        else:
            self.predict_dataset = TextIterDataset(predict_dataset)
        self.test_dataset = ConlluMapDataset(
            test_dataset,
            reverse_edits=self.reverse_edits,
            path_name=path_name,
            train=False,
        )
        with open(f"{path_name}/multiword_dict.json", "r") as mw_tb:
            self.multiword_table = json.load(mw_tb)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.checkpoint = checkpoint
        if self.train_dataset:
            # this is a bit hacky, but not sure how to do this with setup & CLI
            # + 3 is the padding & unk tok & _
            self.pos_classes_cnt = len(self.train_dataset.UPOS_CLASSES) + len(
                SPECIAL_TOKENS
            )
            self.xpos_classes_cnt = len(self.train_dataset.xpos_classes) + len(
                SPECIAL_TOKENS
            )
            self.lemma_classes_cnt = len(
                self.train_dataset.lemma_classes
            ) + len(SPECIAL_TOKENS)
            self.feats_classes_cnt = len(
                self.train_dataset.feats_classes
            ) + len(SPECIAL_TOKENS)
        else:
            self._set_values_from_path_name()

    def _set_values_from_path_name(self):
        assert self.checkpoint, "model checkpoint must not be none"
        hps = torch.load(self.checkpoint)["hyper_parameters"]
        self.pos_classes_cnt = hps.get("pos_out_label_size", 0)
        self.xpos_classes_cnt = hps.get("xpos_out_label_size", 0)
        self.lemma_classes_cnt = hps.get("lemma_out_label_size", 0)
        self.feats_classes_cnt = hps.get("feats_out_label_size", 0)

    def _adjust_sentence(self, sentences, lang_with_space=True):
        delimiter = " " if lang_with_space else ""
        new_sents = []
        replacements = (
            []
        )  # will need to keep track of these to write out multiword tokens
        # preprocessing sentences. The table is usually small so this is not too expensive
        for sentence in sentences:
            new_sentence = sentence
            rep_i = []
            for k, v in self.multiword_table.items():
                if re.search(rf"\b{k}\b", sentence) and v["frequency"] > 2:
                    new_sentence = re.sub(
                        rf"\b{k}\b", delimiter.join(v["words"]), new_sentence
                    )
                    rep_i.append((k, delimiter.join(v["words"])))
            new_sents.append(new_sentence)
            replacements.append(rep_i)
        return new_sents, replacements

    def _pretokenize(self, sentences):
        batch_toks = []
        for sentence in sentences:
            tokens = [ROOT_TOKEN]
            tokens.extend(
                token.text for token in self.pretokenizer(sentence)
            )  # root token is needed for dependency parsing
            batch_toks.append(tokens)
        return batch_toks

    def _collator(self, conllu=True):
        preprocessing = (
            self.conllu_preprocessing
            if conllu
            else self.txt_file_preprocessing
        )
        return partial(
            preprocessing,
            tokenizer=self.tokenizer,
            lang_with_space=self.lang_with_space,
        )

    def conllu_preprocessing(
        self, batch, tokenizer: AutoTokenizer, lang_with_space=True
    ):
        """Data pipeline -> tokenizing input and grouping token IDs to words. The output has varied dimensions"""
        sentences, *args = zip(*batch)
        sentences, replacements = self._adjust_sentence(
            sentences, lang_with_space=lang_with_space
        )
        pretoks = self._pretokenize(sentences)
        # I want to load in tokens by batch to avoid using the inefficient max_length padding
        tokenized_x = tokenizer(
            pretoks,
            padding="longest",
            return_tensors="pt",
            is_split_into_words=True,
        )
        if not tokenized_x.encodings:
            # for Byt5, which doesn't seem to have a fast tokenizer
            encodings_ = []
            for sent in pretoks:
                idxs = []
                toks = []
                for i, word in enumerate(sent):
                    # number of toks
                    toks_ = list(word.encode("utf-8"))
                    num_toks = len(toks_)
                    idxs.extend([i] * num_toks)
                    toks.extend(toks_)
                # this is not the prettiest, but Encoding does not have an __init__(), so this is set manually
                encodings_.append(CustomEncoding(word_ids=idxs, tokens=toks))
            tokenized_x.custom_encodings = encodings_
        return ConlluBatch(tokenized_x, sentences, replacements, *args)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collator(conllu=True),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collator(conllu=True),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collator(conllu=True),
        )

    def txt_file_preprocessing(
        self, sentences, tokenizer: AutoTokenizer, lang_with_space=True
    ):
        """This is the case where the input is NOT a conllu file"""
        sentences, replacements = self._adjust_sentence(
            sentences, lang_with_space=lang_with_space
        )
        pretoks = self._pretokenize(sentences)
        tokenized_x = tokenizer(
            pretoks,
            padding="longest",
            return_tensors="pt",
            is_split_into_words=True,
        )
        return TextBatch(tokenized_x, sentences, replacements)

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collator(conllu=False),
        )
