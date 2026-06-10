"""Logits object."""

import torch
from torch import nn


class Logits(nn.Module):
    """Logits from the classifier forward pass.

    Each tensor is either null or of shape N x C x L."""

    upos: torch.Tensor | None
    xpos: torch.Tensor | None
    lemma: torch.Tensor | None
    feats: torch.Tensor | None
    head: torch.Tensor | None
    deprel: torch.Tensor | None

    def __init__(
        self,
        upos=None,
        xpos=None,
        lemma=None,
        feats=None,
        head=None,
        deprel=None,
    ):
        super().__init__()
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
    def use_parse(self) -> bool:
        return self.head is not None and self.deprel is not None
