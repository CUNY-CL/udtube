"""Encodes and decodes tensors."""

from __future__ import annotations

import dataclasses
from typing import Callable, Iterable, Iterator

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
        deprels: Iterable[str],
        functor: Callable[[str], int],
    ) -> torch.Tensor:
        """Encodes a tensor.

        Args:
            deprels: iterable of deprels.
            functor: a callable mapping from strings to integers; usually
                this is the vocabulary object.

        Returns:
            Tensor of encoded deprels.
        """
        return torch.tensor([functor(deprel) for deprel in deprels])

    def encode_upos(self, tags: Iterable[str]) -> torch.Tensor:
        """Encodes universal POS tags.

        Args:
            tags: iterable of universal POS strings.

        Returns:
            Tensor of encoded tags.
        """
        return self._encode(tags, self.index.upos)

    def encode_xpos(self, tags: Iterable[str]) -> torch.Tensor:
        """Encodes language-specific POS tags.

        Args:
            tags: iterable of language-specific POS strings.

        Returns:
            Tensor of encoded tags.
        """
        return self._encode(tags, self.index.xpos)

    def encode_lemma(
        self, forms: Iterable[str], lemmas: Iterable[str]
    ) -> torch.Tensor:
        """Encodes lemma (i.e., edit script) tags.

        Args:
            forms: iterable of wordforms.
            lemmas: iterable of lemmas.

        Returns:
            Tensor of encoded lemma tags.
        """
        return self._encode(
            [
                self.lemma_mapper.tag(form, lemma)
                for form, lemma in zip(forms, lemmas)
            ],
            self.index.lemma,
        )

    def encode_feats(self, tags: Iterable[str]) -> torch.Tensor:
        """Encodes morphological feature tags.

        Args:
            tags: iterable of feature tags.

        Returns:
            Tensor of encoded features.
        """
        return self._encode(tags, self.index.feats)

    def encode_head(self, indices: Iterable[str]) -> torch.Tensor:
        """Encodes dependency parsing head indices.

        Args:
            indices: iterable of head indices.

        Returns:
            Tensor of encoded head indices.

        """
        # Cheeky, but it works.
        return self._encode(indices, int)

    def encode_deprel(self, deprel: Iterable[str]) -> torch.Tensor:
        """Encodes dependency parsing arc deprels.

        Args:
            deprel: iterable of arc deprels.

        Returns:
            Tensor of encoded arc deprels.
        """
        return self._encode(deprel, self.index.deprel)

    # Decoding.

    @staticmethod
    def _decode(
        indices: torch.Tensor,
        functor: Callable[[int], str],
    ) -> Iterator[str]:
        """Decodes a tensor.

        Args:
            indices: tensor of indices.
            functor: a callable mapping from strings to integers; usually
                this is the vocabulary object's `get_symbol` method.

        Yields:
            Decoded symbols.
        """
        for idx in indices:
            if idx == special.PAD_IDX:
                # To avoid sequence length mismatches,
                # _ is yielded for anything classified as a pad.
                yield special.BLANK
            else:
                yield functor(idx)

    def decode_upos(self, indices: torch.Tensor) -> Iterator[str]:
        """Decodes an upos tensor.

        Args:
            indices: tensor of indices.

        Yields:
            Decoded upos tags.
        """
        return self._decode(indices, self.index.upos.get_symbol)

    def decode_xpos(self, indices: torch.Tensor) -> Iterator[str]:
        """Decodes an xpos tensor.

        Args:
            indices: tensor of indices.

        Yields:
            Decoded xpos tags.
        """
        return self._decode(indices, self.index.xpos.get_symbol)

    def decode_lemma(
        self, forms: Iterable[str], indices: torch.Tensor
    ) -> Iterator[str]:
        """Decodes a lemma tensor.

        Args:
            forms: iterable of wordforms.
            indices: tensor of indices.

        Yields:
            Decoded lemmas.
        """
        for form, tag in zip(
            forms, self._decode(indices, self.index.lemma.get_symbol)
        ):
            yield self.lemma_mapper.lemmatize(form, tag)

    def decode_feats(self, indices: torch.Tensor) -> Iterator[str]:
        """Decodes a morphological features tensor.

        Args:
            indices: tensor of indices.

        Yields:
            Decoded morphological features.
        """
        return self._decode(indices, self.index.feats.get_symbol)

    def decode_head(self, indices: torch.Tensor) -> Iterator[str]:
        """Encodes dependency parsing head indices.

        Args:
            indices: iterable of head indices.

        Returns:
            Decoded head indices, as strings.
        """
        return self._decode(indices, lambda idx: str(idx.item()))

    def decode_deprel(self, indices: torch.Tensor) -> Iterator[str]:
        """Decodes dependency parsing arc deprels.

        Args:
            indices: tensor of indices.

        Yields:
            Decoded arc deprels.
        """
        return self._decode(indices, self.index.deprel.get_symbol)
