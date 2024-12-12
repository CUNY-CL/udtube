"""Collators."""

from typing import List

import torch
from torch import nn
import transformers

from .. import special
from . import batches, datasets


class Tokenizer:
    """Helper for tokenization.

    Args:
        tokenizer: transformer autotokenizer.
    """

    tokenizer: transformers.AutoTokenizer

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, tokens: List[List[str]]) -> transformers.BatchEncoding:
        """Runs tokenizer.

        Args:
            tokens: a list of sentences, each a list of tokens.

        Returns:
            The BatchEncoding object.
        """
        return self.tokenizer(
            tokens,
            padding="longest",
            return_tensors="pt",
            is_split_into_words=True,
            add_special_tokens=False,
        )


class Collator:
    """Collator for CoNLL-U data.

    It doesn't matter whether or not the data already has labels.

    Args:
        tokenizer: transformer autotokenizer.
    """

    def __init__(self, tokenizer: transformers.AutoTokenizer):
        self.tokenizer = Tokenizer(tokenizer)

    @staticmethod
    def pad_tensors(
        tensorlist: List[torch.Tensor],
    ) -> torch.Tensor:
        """Pads and stacks a list of tensors.

        Args:
            tensorlist: a list of tensors to be padded.

        Returns:
            The padded and stacked tensor.
        """
        pad_max = max(len(tensor) for tensor in tensorlist)
        return torch.stack(
            [
                nn.functional.pad(
                    tensor, (0, pad_max - len(tensor)), value=special.PAD_IDX
                )
                for tensor in tensorlist
            ]
        )

    def __call__(self, itemlist: List[datasets.Item]) -> batches.Batch:
        return batches.Batch(
            tokenlists=[item.tokenlist for item in itemlist],
            tokens=self.tokenizer([item.tokens for item in itemlist]),
            # Looks ugly, but this just pads and stacks data for whatever
            # classification tasks are enabled.
            upos=(
                self.pad_tensors([item.upos for item in itemlist])
                if itemlist[0].use_upos
                else None
            ),
            xpos=(
                self.pad_tensors([item.xpos for item in itemlist])
                if itemlist[0].use_xpos
                else None
            ),
            lemma=(
                self.pad_tensors([item.lemma for item in itemlist])
                if itemlist[0].use_lemma
                else None
            ),
            feats=(
                self.pad_tensors([item.feats for item in itemlist])
                if itemlist[0].use_feats
                else None
            ),
        )
