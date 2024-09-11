import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
import logging

from typing import List, Dict, Tuple, Union, TextIO, Iterator

import conllu
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch import tensor
from torch.utils.data import Dataset, IterableDataset

import edit_scripts
from defaults import OVERRIDDEN_FIELD_PARSERS, UNK_TAG, PAD_TAG


class FileNameError(Exception):
    pass


class ConlluMapDataset(Dataset):
    """Conllu Map Dataset, used for small training/validation sets

    This class loads the entire dataset into memory and is therefore only suitable for smaller datasets
    """

    UPOS_CLASSES = [
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
    ]

    def __init__(
        self,
        conllu_file: str,
        reverse_edits: bool = False,
        path_name: str = "UDTube",
        train: bool = False,
    ):
        """Initializes the instance based on user input.

        Args:
            conllu_file: The path to the conllu file to make the dataset from
            reverse_edits: Reverse edit script calculation. Recommended for suffixal languages. False by default
            path_name: The dir where everything will get saved
            train: Should only be true for the train dataset. Used to decide if to fit encoders
        """
        super().__init__()
        self.path_name = path_name
        self.all_words = []
        self.conllu_file = conllu_file
        self.e_script = (
            edit_scripts.ReverseEditScript
            if reverse_edits
            else edit_scripts.EditScript
        )
        # setting up label encoders
        if conllu_file:
            self.xpos_classes = self._get_all_classes("xpos")
            self.feats_classes = self._get_all_classes("feats")
            self.lemma_classes = self._get_all_classes("lemma")
            if train:
                self.upos_encoder = LabelEncoder()
                self.xpos_encoder = LabelEncoder()
                self.ufeats_encoder = LabelEncoder()
                self.lemma_encoder = LabelEncoder()
                self._fit_label_encoders()
                self.multiword_table = {}
            else:
                self._load_label_encoders()
            self.data_set = self._get_data(train)
        else:
            # Instantiation of empty class, happens in prediction
            self.data_set = []

    def _fit_label_encoders(self) -> None:
        self.upos_encoder.fit(self.UPOS_CLASSES + [PAD_TAG, UNK_TAG, "_"])
        self.xpos_encoder.fit(self.xpos_classes + [PAD_TAG, UNK_TAG, "_"])
        self.ufeats_encoder.fit(self.feats_classes + [PAD_TAG, UNK_TAG, "_"])
        self.lemma_encoder.fit(self.lemma_classes + [PAD_TAG, UNK_TAG, "_"])
        # saving all the encoders
        try:
            os.mkdir(self.path_name)
        except FileExistsError:
            # dir already exists
            pass
        logging.info(f"Now saving label encoders for {self.conllu_file}")
        joblib.dump(self.upos_encoder, f"{self.path_name}/upos_encoder.joblib")
        joblib.dump(self.xpos_encoder, f"{self.path_name}/xpos_encoder.joblib")
        joblib.dump(
            self.ufeats_encoder, f"{self.path_name}/ufeats_encoder.joblib"
        )
        joblib.dump(
            self.lemma_encoder, f"{self.path_name}/lemma_encoder.joblib"
        )

    def _load_label_encoders(self):
        self.lemma_encoder = joblib.load(
            f"{self.path_name}/lemma_encoder.joblib"
        )
        self.ufeats_encoder = joblib.load(
            f"{self.path_name}/ufeats_encoder.joblib"
        )
        self.upos_encoder = joblib.load(
            f"{self.path_name}/upos_encoder.joblib"
        )
        self.xpos_encoder = joblib.load(
            f"{self.path_name}/xpos_encoder.joblib"
        )

    def _get_all_classes(
        self, lname: str
    ) -> List[Union[str, None, int, List[int], Dict[str, str]]]:
        """helper function to get all the classes observed in the training set"""
        classes = []
        with open(self.conllu_file) as f:
            dt = conllu.parse_incr(f, field_parsers=OVERRIDDEN_FIELD_PARSERS)
            for tk_list in dt:
                for tok in tk_list:
                    item = tok[lname] if tok[lname] else "_"
                    if lname != "lemma" and item not in classes:
                        classes.append(item)
                    elif lname == "lemma":
                        lrule = str(
                            self.e_script(tok["form"].lower(), item.lower())
                        )
                        if lrule not in classes:
                            classes.append(lrule)
        return classes

    def _get_data(self, train: bool) -> List[Tuple[int]]:
        """Loads in the conllu data and prepares it as a list dataset so that it can be indexed for __getitem__"""
        data = []
        with open(self.conllu_file) as f:
            dt = conllu.parse_incr(f, field_parsers=OVERRIDDEN_FIELD_PARSERS)
            for tk_list in dt:
                sentence = tk_list.metadata["text"]
                uposes = []
                xposes = []
                lemma_rules = []
                ufeats = []
                for i, tok in enumerate(tk_list):
                    if isinstance(tok["id"], tuple):
                        if train:
                            # This is a multi-word token, id looks like (1, '-', 3)
                            # creating a look-up table to be used when loading in sentences
                            start, sep, end = tok[
                                "id"
                            ]  # these are not perfect indices because of the multiword token stuff
                            if not sep == ".":
                                # when sep is ., this is an ellided token, not a multiword token
                                x = (end - start) + i
                                # below, i + 1 is because we want the next token, not the current
                                words = [
                                    t["form"] for t in tk_list[i + 1 : x + 2]
                                ]
                                if tok["form"] in self.multiword_table:
                                    self.multiword_table[tok["form"]][
                                        "frequency"
                                    ] += 1
                                else:
                                    self.multiword_table[tok["form"]] = {
                                        "frequency": 1,
                                        "words": words,
                                    }
                        continue  # we don't want to add it to the dataset!
                    l_rule = str(
                        self.e_script(
                            tok["form"].lower(), tok["lemma"].lower()
                        )
                    )
                    # Label encoders can't handle None!
                    if tok["feats"] is None:
                        tok["feats"] = "_"

                    # Here we have to check if the label in the dataset is unknown
                    upos_ = (
                        tok["upos"]
                        if tok["upos"] in self.upos_encoder.classes_
                        else UNK_TAG
                    )
                    xpos_ = (
                        tok["xpos"]
                        if tok["xpos"] in self.xpos_encoder.classes_
                        else UNK_TAG
                    )
                    ufeat_ = (
                        tok["feats"]
                        if tok["feats"] in self.ufeats_encoder.classes_
                        else UNK_TAG
                    )
                    if l_rule not in self.lemma_encoder.classes_:
                        l_rule = UNK_TAG
                    uposes.append(upos_)
                    xposes.append(xpos_)
                    lemma_rules.append(l_rule)
                    ufeats.append(ufeat_)
                    self.all_words.append(tok["form"].lower())
                uposes = self.upos_encoder.transform(uposes)
                xposes = self.xpos_encoder.transform(xposes)
                lemma_rules = self.lemma_encoder.transform(lemma_rules)
                ufeats = self.ufeats_encoder.transform(ufeats)
                # heads do not need to be encoded, they are already numerical. Encoding them here causes problems later
                data.append((sentence, uposes, xposes, lemma_rules, ufeats))
            if train:
                self.check_multiword_table()
                with open(
                    f"{self.path_name}/multiword_dict.json", "w"
                ) as mw_tb:
                    json.dump(self.multiword_table, mw_tb, ensure_ascii=False)
        return data

    def check_multiword_table(self) -> None:
        """Make sure all items are valid. Some entries could be typos or annotator mistakes,
        if it is more frequent as a single word, it is assumed to be a typo/annotator mistake.
        """
        single_cnt = Counter(self.all_words)
        keep = {}
        for w, record in self.multiword_table.items():
            if w in single_cnt and single_cnt[w] > record["frequency"]:
                continue
            keep[w] = record
        self.multiword_table = keep

    def get_special_words(self) -> List[str]:
        """Special words are ones that are not handled well by tokenizers, i.e. 2000-2004 or K., which are considered
        single tokens in some conllu files."""
        special_words = []
        with open(self.conllu_file) as f:
            dt = conllu.parse_incr(f, field_parsers=OVERRIDDEN_FIELD_PARSERS)
            for tk_list in dt:
                for tk in tk_list:
                    if isinstance(tk["id"], int) and re.search(
                        r"\w+[.'-\\/%]|[.'-\\/%]\w+", tk["form"]
                    ):
                        special_words.append(tk["form"])
        return special_words

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]


class TextIterDataset(IterableDataset):
    """Iterable dataset for non-conllu files, used for inference when labels are unknown

    This class is used when the data for inference is large, and we do not want to load it entirely into memory.
    """

    def __init__(self, text_file: str):
        """Initializes the instance based on user input.

        Args:
            text_file: The path to the non-conllu textfile to incrementally read from.
        """
        super().__init__()
        self.tf = text_file

    def __iter__(self) -> TextIO:
        if isinstance(self.tf, str):
            self.tf = open(self.tf)
        return self.tf

    def __del__(self) -> None:
        if hasattr(self, "tf") and self.tf:
            self.tf.close()  # want to make sure we close the file during garbage collection


class ConlluIterDataset(IterableDataset):
    def __init__(self, conllu_file: str):
        """Initializes the instance based on user input.

        Args:
            conllu_file: The path to the conllu textfile to incrementally read from.
        """
        super().__init__()
        self.conllu_file = conllu_file

    def __iter__(self) -> Iterator[str]:
        if isinstance(self.conllu_file, str):
            self.conllu_file = open(self.conllu_file)
        parsed_conllu_generator = conllu.parse_incr(
            self.conllu_file, field_parsers=OVERRIDDEN_FIELD_PARSERS
        )
        return (
            token_list.metadata["text"]
            for token_list in parsed_conllu_generator
        )

    def __del__(self) -> None:
        if not isinstance(self.conllu_file, str):
            self.conllu_file.close()
