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
        strings: Iterable[str],
        functor: Callable[[str], int],
    ) -> torch.Tensor:
        """Encodes a tensor.

        Args:
            strings: iterable of strings.
            functor: a callable mapping from strings to integers; usually
                this is the vocabulary object.

        Returns:
            Tensor of encoded strings
        """
        return torch.tensor([functor(string) for string in strings])

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
        return self._encode(indices, lambda idx: int(idx) + special.OFFSET)

    def encode_deprel(self, deprel: Iterable[str]) -> torch.Tensor:
        """Encodes dependency parsing dependency relations.

        Args:
            deprel: iterable of dependency relations.

        Returns:
            Tensor of encoded dependency relations.
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
            # FIXME(kbg): is this the right thing to do? Do I need a special
            # case for padding?
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
            Decoded head indices.
        """
        return self._decode(
            indices, lambda idx: str(idx.item() - special.OFFSET)
        )

    def decode_deprel(self, indices: torch.Tensor) -> Iterator[str]:
        """Decodes dependency parsing dependency relations.

        Args:
            indices: tensor of indices.

        Yields:
            Decoded dependency relations.
        """
        return self._decode(indices, self.index.deprel.get_symbol)
