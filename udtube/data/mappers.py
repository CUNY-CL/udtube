"""Encodes and decodes tensors."""

from __future__ import annotations

import dataclasses
from typing import Iterable, Iterator

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
    ) -> Iterator[str]:
        """Decodes a tensor.

        Args:
            indices: tensor of indices.
            vocabulary: the vocabulary

        Yields:
            str: decoded symbols.
        """
        for idx in indices:
            if idx == special.PAD_IDX:
                # To avoid sequence length mismatches,
                # _ is yielded for anything classified as a pad.
                yield "_"
            else:
                yield vocabulary.get_symbol(idx)

    def decode_upos(self, indices: torch.Tensor) -> Iterator[str]:
        """Decodes an upos tensor.

        Args:
            indices: tensor of indices.

        Yields:
            str: decoded upos tags.
        """
        return self._decode(indices, self.index.upos)

    def decode_xpos(self, indices: torch.Tensor) -> Iterator[str]:
        """Decodes an xpos tensor.

        Args:
            indices: tensor of indices.

        Yields:
            str: decoded xpos tags.
        """
        return self._decode(indices, self.index.xpos)

    def decode_lemma(
        self, forms: Iterable[str], indices: torch.Tensor
    ) -> Iterator[str]:
        """Decodes a lemma tensor.

        Args:
            forms: iterable of wordforms.
            indices: tensor of indices.

        Yields:
            str: decoded lemmas.
        """
        for form, tag in zip(forms, self._decode(indices, self.index.lemma)):
            yield self.lemma_mapper.lemmatize(form, tag)

    def decode_feats(self, indices: torch.Tensor) -> Iterator[str]:
        """Decodes a morphological features tensor.

        Args:
            indices: tensor of indices.

        Yields:
            str: decoded morphological features.
        """
        return self._decode(indices, self.index.feats)
