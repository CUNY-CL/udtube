"""Encodes and decodes tensors."""

from __future__ import annotations

import dataclasses
from typing import Iterable, List

import torch

from . import edit_scripts, indexes
from .. import defaults, special


@dataclasses.dataclass
class LemmaMapper:
    """Handles lemmatization rules."""

    reverse_edits: bool = defaults.REVERSE_EDITS

    @property
    def edit_script(self) -> edit_scripts.EditScript:
        return (
            edit_scripts.ReverseEditScript
            if self.reverse_edits
            else edit_scripts.EditScript
        )

    def tag(self, form: str, lemma: str) -> str:
        """Computes the lemma tag."""
        return str(self.edit_script(form.casefold(), lemma.casefold()))

    def lemmatize(self, form: str, tag: str) -> str:
        """Applies the lemma tag to a form."""
        rule = self.edit_script.fromtag(tag)
        return rule.apply(form.casefold())


@dataclasses.dataclass
class Mapper:
    """Handles mapping between strings and tensors."""

    index: indexes.Index  # Usually copied from the DataModule.

    def __post_init__(self):
        self.lemma_mapper = LemmaMapper(self.index.reverse_edits)

    @classmethod
    def read(cls, model_dir: str) -> Mapper:
        """Loads mapper from an index.

        Args:
            model_dir (str).

        Returns:
            Mapper.
        """
        return cls(indexes.Index.read(model_dir))

    # Encoding.

    @staticmethod
    def _encode(
        labels: Iterable[str],
        vocabulary: indexes.Vocabulary,
    ) -> torch.Tensor:
        """Encodes a tensor.

        Args:
            labels: iterable of labels.
            vocabulary: a vocabulary.

        Returns:
            Tensor of encoded labels.
        """
        return torch.tensor([vocabulary(label) for label in labels])

    def encode_upos(self, labels: Iterable[str]) -> torch.Tensor:
        """Encodes universal POS tags.

        Args:
            labels: iterable of universal POS strings.

        Returns:
            Tensor of encoded labels.
        """
        return self._encode(labels, self.index.upos)

    def encode_xpos(self, labels: Iterable[str]) -> torch.Tensor:
        """Encodes language-specific POS tags.

        Args:
            labels: iterable of label-specific POS strings.

        Returns:
            Tensor of encoded labels.
        """
        return self._encode(labels, self.index.xpos)

    def encode_lemma(
        self, forms: Iterable[str], lemmas: Iterable[str]
    ) -> torch.Tensor:
        """Encodes lemma (i.e., edit script) tags.

        Args:
            forms: iterable of wordforms.
            lemmas: iterable of lemmas.

        Returns:
            Tensor of encoded labels.
        """
        return self._encode(
            [
                self.lemma_mapper.tag(form, lemma)
                for form, lemma in zip(forms, lemmas)
            ],
            self.index.lemma,
        )

    def encode_feats(self, labels: Iterable[str]) -> torch.Tensor:
        """Encodes morphological feature tags.

        Args:
            labels: iterable of feature tags.

        Returns:
            Tensor of encoded labels.
        """
        return self._encode(labels, self.index.feats)

    # Decoding.

    @staticmethod
    def _decode(
        indices: torch.Tensor,
        vocabulary: indexes.Vocabulary,
    ) -> List[str]:
        """Decodes a tensor.

        Args:
            indices: tensor of indices.
            vocabulary: the vocabulary

        Yields:
            List[str]: decoded symbols.
        """
        symbols = []
        for idx in indices:
            if idx == special.PAD_IDX:
                return symbols
            symbols.append(vocabulary.get_symbol(idx))
        return symbols

    def decode_upos(self, indices: torch.Tensor) -> List[str]:
        """Decodes an upos tensor.

        Args:
            indices: tensor of indices.

        Yields:
            List[str]: decoded upos tags.
        """
        return self._decode(indices, self.index.upos)

    def decode_xpos(self, indices: torch.Tensor) -> List[str]:
        """Decodes an xpos tensor.

        Args:
            indices: tensor of indices.

        Yields:
            List[str]: decoded xpos tags.
        """
        return self._decode(indices, self.index.xpos)

    def decode_lemma(
        self, forms: Iterable[str], indices: torch.Tensor
    ) -> List[str]:
        """Decodes a lemma tensor.

        Args:
            forms: iterable of wordforms.
            indices: tensor of indices.

        Yields:
            List[str]: decoded lemmas.
        """
        return [
            self.lemma_mapper.lemmatize(form, tag)
            for form, tag in zip(
                forms, self._decode(indices, self.index.lemma)
            )
        ]

    def decode_feats(self, indices: torch.Tensor) -> List[str]:
        """Decodes a morphological features tensor.

        Args:
            indices: tensor of indices.

        Yields:
            List[str]: decoded morphological features.
        """
        return self._decode(indices, self.index.feats)
