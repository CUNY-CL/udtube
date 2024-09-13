"""Data modules."""

import os
from typing import Optional

import lightning
import transformers
from torch.utils import data

from .. import defaults
from . import collators, datasets, indexes, mappers, parsers


class Error(Exception):
    pass


class DataModule(lightning.LightningDataModule):
    """CoNLL-U data module.

    This class is initialized by the LightningCLI interface. It manages all
    data loading steps for UDTube. Training and validation are loaded into
    fully into memory; prediction data is loaded incrementally to avoid memory
    errors with large datasets.

    Args:
        model_dir: path for checkpoints, indexes, and logs.
        predict: path to the prediction CoNLL-U file.
        test: path to the test CoNLL-U file.
        train: path to training CoNLL-U file.
        val: path to validation CoNLL-U file.
        encoder: name of encoder on Hugging Face (for the tokenizer).
        reverse_edits: enables reverse (suffixal) edit scripts.
        use_upos: enables the universal POS tagging task.
        use_xpos: enables the language-specific POS tagging task.
        use_lemma: enables the lemmatization task.
        use_feats: enables the morphological feature tagging task.
        batch_size: batch size.
        index: an optional pre-computed index.
    """

    predict: Optional[str]
    test: Optional[str]
    train: Optional[str]
    val: Optional[str]
    reverse_edits: bool
    use_upos: bool
    use_xpos: bool
    use_lemma: bool
    use_feats: bool
    batch_size: int
    index: indexes.Index
    tokenizer: transformers.AutoTokenizer

    def __init__(
        self,
        # Paths.
        *,
        model_dir: str,
        test=None,
        train=None,
        predict=None,
        val=None,
        # Modeling options.
        encoder: str = defaults.ENCODER,
        reverse_edits: bool = defaults.REVERSE_EDITS,
        use_upos: bool = defaults.USE_UPOS,
        use_xpos: bool = defaults.USE_XPOS,
        use_lemma: bool = defaults.USE_LEMMA,
        use_feats: bool = defaults.USE_FEATS,
        # Other.
        batch_size: int = defaults.BATCH_SIZE,
        index: Optional[indexes.Index] = None,
    ):
        super().__init__()
        self.train = train
        self.val = val
        self.predict = predict
        self.test = test
        self.reverse_edits = reverse_edits
        self.use_upos = use_upos
        self.use_xpos = use_xpos
        self.use_lemma = use_lemma
        self.use_feats = use_feats
        self.batch_size = batch_size
        # FIXME I need to be able to load the index from a file too.
        self.index = index if index else self._make_index(model_dir)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(encoder)

    # Based on: https://universaldependencies.org/u/pos/index.html.

    UPOS_VOCABULARY = [
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

    def _make_index(self, model_dir: str) -> indexes.Index:
        xpos_vocabulary = set() if self.use_xpos else None
        lemma_vocabulary = set() if self.use_lemma else None
        feats_vocabulary = set() if self.use_feats else None
        lemma_mapper = mappers.LemmaMapper(self.reverse_edits)
        for tokenlist in parsers.parse(self.train):
            # We don't need to collect the upos vocabulary because "u" stands
            # for "universal" here.
            if self.use_xpos:
                xpos_vocabulary.update(token["xpos"] for token in tokenlist)
            if self.use_lemma:
                for token in tokenlist:
                    lemma_vocabulary.add(
                        lemma_mapper.tag(token["form"], token["lemma"])
                    )
            if self.use_feats:
                feats_vocabulary.update(token["feats"] for token in tokenlist)
        result = indexes.Index(
            reverse_edits=self.reverse_edits,
            upos=(
                indexes.Vocabulary(self.UPOS_VOCABULARY)
                if self.use_upos
                else None
            ),
            xpos=(
                indexes.Vocabulary(xpos_vocabulary) if self.use_xpos else None
            ),
            lemma=(
                indexes.Vocabulary(lemma_vocabulary)
                if self.use_lemma
                else None
            ),
            feats=(
                indexes.Vocabulary(feats_vocabulary)
                if self.use_feats
                else None
            ),
        )
        # Writes it to the model directory.
        try:
            os.mkdir(model_dir)
        except FileExistsError:
            pass
        result.write(model_dir)
        return result

    # Properties.

    @property
    def upos_tagset_size(self) -> int:
        return len(self.index.upos) if self.use_upos else 0

    @property
    def xpos_tagset_size(self) -> int:
        return len(self.index.xpos) if self.use_xpos else 0

    @property
    def lemma_tagset_size(self) -> int:
        return len(self.index.lemma) if self.use_lemma else 0

    @property
    def feats_tagset_size(self) -> int:
        return len(self.index.feats) if self.use_feats else 0

    # Required API.

    def train_dataloader(self) -> data.DataLoader:
        assert self.train is not None, "no train path"
        return data.DataLoader(
            self._conllu_map_dataset(self.train),
            collate_fn=collators.Collator(
                self.tokenizer,
                self.index,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
        )

    def val_dataloader(self) -> data.DataLoader:
        assert self.train is not None, "no val path"
        return data.DataLoader(
            self._conllu_map_dataset(self.val),
            collate_fn=collators.Collator(
                self.tokenizer,
                self.index,
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
        )

    def predict_dataloader(self) -> data.DataLoader:
        assert self.predict is not None, "no predict path"
        return data.DataLoader(
            # This one uses an iterative data loader instead.
            datasets.ConlluIterDataset(self.predict),
            collate_fn=collators.Collator(
                self.tokenizer,
                self.index,
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
        )

    def test_dataloader(self) -> data.DataLoader:
        assert self.test is not None, "no test path"
        return data.DataLoader(
            self._conllu_map_dataset(self.test),
            collate_fn=collators.Collator(
                self.tokenizer,
                self.index,
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
        )

    def _conllu_map_dataset(self, source: str) -> datasets.ConlluMapDataset:
        return datasets.ConlluMapDataset(
            list(parsers.parse(source)),
            mappers.Mapper(self.index),
            self.use_upos,
            self.use_xpos,
            self.use_lemma,
            self.use_feats,
        )
