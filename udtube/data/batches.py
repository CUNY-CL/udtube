"""Batch objects."""

from typing import List, Optional

import tokenizers
import torch
from torch import nn

from . import conllu


class Batch(nn.Module):
    """CoNLL-U data batch.

    This can handle padded label tensors if present.

    Args:
        tokenlists: list of TokenLists.
        tokens: batch encoding from the transformer.
        pos: optional padded tensor of universal POS labels.
        xpos: optional padded tensor of language-specific POS labels.
        lemma: optional padded tensor of lemma labels.
        feats: optional padded tensor of morphological feature labels.
    """

    tokenlists: List[conllu.TokenList]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    encodings: List[tokenizers.Encoding]
    upos: Optional[torch.Tensor]
    xpos: Optional[torch.Tensor]
    lemma: Optional[torch.Tensor]
    feats: Optional[torch.Tensor]

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

    def __len__(self) -> int:
        return len(self.tokenlists)
