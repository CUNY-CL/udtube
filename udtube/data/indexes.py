"""Symbol indexes.

Adapted from Yoyodyne:

    https://github.com/CUNY-CL/yoyodyne/blob/master/yoyodyne/data/indexes.py

The major difference here is that we have a separate vocabulary for each
classifier layer, since unlike in Yoyodyne, there is no sense in which we
we could share a vocabulary or an embedding across classifier layers.

Because of this, we also have the Index class, which holds instances of
the lower-level vocabularies, one for each enabled classifier head, and which
handles (de)serialization."""

from __future__ import annotations

import dataclasses
import pickle
from typing import Dict, Iterable, List, Optional

from .. import defaults, special


class Vocabulary:
    """Maintains an index over a vocabulary."""

    _index2symbol: List[str]
    _symbol2index: Dict[str, int]

    def __init__(self, vocabulary: Iterable[str]):
        # TODO: consider storing this in-class for logging purposes.
        self._index2symbol = special.SPECIAL + sorted(vocabulary)
        self._symbol2index = {c: i for i, c in enumerate(self._index2symbol)}

    def __len__(self) -> int:
        return len(self._index2symbol)

    # Lookup.

    def __call__(self, lookup: str) -> int:
        """Looks up index by symbol.

        Args:
            symbol (str).

        Returns:
            int.
        """
        return self._symbol2index.get(lookup, special.UNK_IDX)

    def get_symbol(self, index: int) -> str:
        """Looks up symbol by index.

        Args:
            index (int).

        Returns:
            str.
        """
        return self._index2symbol[index]


@dataclasses.dataclass
class Index:
    """A collection of vocabularies, one per enabled classification task.

    This also handles serialization and deserialization.

    Args:
        reverse_edits:
        upos: optional vocabulary for universal POS tagging.
        xpos: optional vocabulary for language-specific POS tagging.
        lemma: optional vocabulary for lemmatization.
        feats: optional vocabulary for morphological tagging.
    """

    reverse_edits: bool = defaults.REVERSE_EDITS
    upos: Optional[Vocabulary] = None
    xpos: Optional[Vocabulary] = None
    lemma: Optional[Vocabulary] = None
    feats: Optional[Vocabulary] = None

    # Serialization.

    @classmethod
    def read(cls, model_dir: str) -> Index:
        """Loads index.

        Args:
            model_dir (str).

        Returns:
            Index.
        """
        index = cls.__new__(cls)
        with open(cls.path(model_dir), "rb") as source:
            for key, value in pickle.load(source).items():
                setattr(index, key, value)
        return index

    def write(self, model_dir: str) -> None:
        with open(self.path(model_dir), "wb") as sink:
            pickle.dump(vars(self), sink)

    @staticmethod
    def path(model_dir: str) -> str:
        return f"{model_dir}/index.pkl"
