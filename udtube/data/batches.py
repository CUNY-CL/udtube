"""Batch objects."""

from typing import List, Optional

import tokenizers
import torch
from torch import nn

from . import conllu


class Batch(nn.Module):
    """CoNLL-U data batch.

    Args:
        tokenlists: list of TokenLists.
        tokens: batch encoding from the transformer.
        pos: optional padded tensor of universal POS deprels.
        xpos: optional padded tensor of language-specific POS deprels.
        lemma: optional padded tensor of lemma deprels.
        feats: optional padded tensor of morphological feature deprels.
        head: optional padded tensor of dependency parser head indices.
        deprel: optional padded tensor of dependency parser arc deprels.
    """

    tokenlists: List[conllu.TokenList]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    encodings: List[tokenizers.Encoding]
    upos: Optional[torch.Tensor]
    xpos: Optional[torch.Tensor]
    lemma: Optional[torch.Tensor]
    feats: Optional[torch.Tensor]
    head: Optional[torch.Tensor]
    deprel: Optional[torch.Tensor]

    def __init__(
        self,
        tokenlists,
        input_ids,
        attention_mask,
        encodings,
        upos=None,
        xpos=None,
        lemma=None,
        feats=None,
        head=None,
        deprel=None,
    ):
        super().__init__()
        self.tokenlists = tokenlists
        self.register_buffer("input_ids", input_ids)
        self.register_buffer("attention_mask", attention_mask)
        self.encodings = encodings
        self.register_buffer("upos", upos)
        self.register_buffer("xpos", xpos)
        self.register_buffer("lemma", lemma)
        self.register_buffer("feats", feats)
        self.register_buffer("head", head)
        self.register_buffer("deprel", deprel)

    @property
    def use_upos(self) -> bool:
        return self.upos is not None

    @property
    def use_xpos(self) -> bool:
        return self.xpos is not None

    @property
    def use_lemma(self) -> bool:
        return self.lemma is not None

    @property
    def use_feats(self) -> bool:
        return self.feats is not None

    @property
    def use_prase(self) -> bool:
        return self.head is not None and self.deprel is not None

    def __len__(self) -> int:
        return len(self.tokenlists)
