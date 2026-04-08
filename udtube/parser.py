"""Biaffine attention dependency parser.

Based on:

    Dozat, T., and Manning, C. D. 2017. Deep biaffine attention for dependency
    parsing. In ICLR.
"""

from typing import Tuple

import torch
from torch import nn

from . import defaults, special


class BiaffineAttention(nn.Module):
    r"""Biaffine attention mechanism for scoring head-dependent pairs.

    This implements the biaffine transformation:

        score(i, j) = h_j^T U h_i + (h_j \oplus h_i)^T w + b

    where h_i is the dependent representation and h_j is the head
    representation.

    Args:
        head_size: Size of head representation.
        dep_size: Size of dependency representation.
        out_size: Output dimension; use 1 for head scores and the number of
            unique depependency relations for deprel scores.
    """

    head_size: int
    dep_size: int
    out_size: int
    weight: nn.Parameter

    def __init__(
        self,
        head_size: int,
        dep_size: int,
        out_size: int = 1,
    ):
        super().__init__()
        self.head_size = head_size
        self.dep_size = dep_size
        self.out_size = out_size
        self.weight = nn.Parameter(
            torch.zeros(self.out_size, self.head_size + 1, self.dep_size + 1)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        head: torch.Tensor,
        dep: torch.Tensor,
    ) -> torch.Tensor:
        """Computes biaffine attention scores.

        Args:
            head: Head representations.
            dep: Dependent representations.

        Returns:
            Scores for each dependent position, scores for all possible heads.
        """
        assert head.shape[0] == dep.shape[0], "Batch size mismatch"
        assert head.shape[1] == dep.shape[1], "Sequence length mismatch"
        assert (
            head.shape[2] == self.head_size
        ), f"Head size mismatch: {head.shape[2]} != {self.head_size}"
        assert (
            dep.shape[2] == self.dep_size
        ), f"Dep size mismatch: {dep.shape[2]} != {self.dep_size}"
        # FIXME =2?
        head = torch.cat((head, torch.ones_like(head[..., :1])), dim=-1)
        # FIXME =2?
        dep = torch.cat((dep, torch.ones_like(dep[..., :1])), dim=-1)
        dep_weight = torch.einsum("bld,odh->blho", dep, self.weight)
        return torch.einsum("bsh,bdho->bdso", head, dep_weight)


class BiaffineParser(nn.Module):
    """Biaffine parser for dependency arc and deprel prediction.

    This takes the encoder outputs and predicts:

    * Head indices for each token.
    * Dependency dependency relations for each arc.

    Following Dozat & Manning, we apply separate MLPs to reduce dimensionality
    before the biaffine classifiers.

    Head data is interpreted as indices, but these indices can collide with
    the 0 used for padding. We therefore shift and unshift the data to avoid
    this collision.

    Args:
        arc_mlp_size: Hidden layer size for arc MLP.
        deprel_mlp_size: Hidden layer size for deprel MLP.
        dropout: Dropout probability for MLP layers
    """

    arc_head_mlp: nn.Module
    arc_dep_mlp: nn.Module
    arc_deprel_head_mlp: nn.Module
    arc_deprel_dep_mlp: nn.Module
    arc_attention: BiaffineAttention
    deprel_attention: BiaffineAttention
    loss_func: nn.CrossEntropyLoss

    def __init__(
        self,
        hidden_size,
        arc_mlp_size: int = defaults.ARC_MLP_SIZE,
        deprel_mlp_size: int = defaults.DEPREL_MLP_SIZE,
        num_deprel: int = 2,  # Dummy value filled in via link.
        dropout: float = defaults.DROPOUT,
    ):
        super().__init__()
        self.arc_head_mlp = self._make_mlp(hidden_size, arc_mlp_size, dropout)
        self.arc_dep_mlp = self._make_mlp(hidden_size, arc_mlp_size, dropout)
        self.deprel_head_mlp = self._make_mlp(
            hidden_size, deprel_mlp_size, dropout
        )
        self.deprel_dep_mlp = self._make_mlp(
            hidden_size, deprel_mlp_size, dropout
        )
        self.arc_attention = BiaffineAttention(
            arc_mlp_size,
            arc_mlp_size,
            1,
        )
        self.deprel_attention = BiaffineAttention(
            deprel_mlp_size,
            deprel_mlp_size,
            num_deprel,
        )
        self.loss_func = nn.CrossEntropyLoss(ignore_index=special.PAD_IDX)

    @staticmethod
    def _make_mlp(
        input_size: int, hidden_size: int, dropout: float
    ) -> nn.Module:
        """Build a single-layer MLP with ReLU activation and dropout.

        Args:
            input_size: Input size.
            hidden_size: Hidden/output size.
            dropout: Dropout probability.

        Returns:
            A sequential MLP module.
        """
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _shift_head(head: torch.Tensor) -> torch.Tensor:
        """Converts indices to internal representation."""
        return torch.where(
            head == special.PAD_IDX,
            head,
            head + special.OFFSET,
        )

    @staticmethod
    def _unshift_head(head: torch.Tensor) -> torch.Tensor:
        """Converts internal representation to indices."""
        return torch.where(
            head == special.PAD_IDX,
            head,
            head - special.OFFSET,
        )

    def forward(
        self,
        encodings: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for dependency parsing.

        Args:
            encodings: Encoder output.
            mask: Attention mask of shape N x L.

        Returns:
            A (arc logits, deprel logits) tuple.
        """
        batch_size = encodings.size(0)
        length = encodings.size(1)
        arc_head = self.arc_head_mlp(encodings)
        arc_dep = self.arc_dep_mlp(encodings)
        deprel_head = self.deprel_head_mlp(encodings)
        deprel_dep = self.deprel_dep_mlp(encodings)
        # FIXME indices.
        arc_logits = self.arc_attention(arc_head, arc_dep).squeeze(-1)
        deprel_logits = self.deprel_attention(deprel_head, deprel_dep)
        arc_mask = mask.unsqueeze(1)
        arc_logits.masked_fill_(~arc_mask, defaults.NEG_EPSILON)
        arc_mask = arc_mask.unsqueeze(-1)
        deprel_logits.masked_fill_(~arc_mask, defaults.NEG_EPSILON)
        assert arc_logits.shape == (
            batch_size,
            length,
            length,
        ), f"Arc logits shape mismatch: {arc_logits.shape}"
        assert deprel_logits.shape == (
            batch_size,
            length,
            length,
            self.deprel_attention.out_size,
        ), f"Deprel logits shape mismatch: {deprel_logits.shape}"
        return arc_logits, deprel_logits

    def compute_loss(
        self,
        head_logits: torch.Tensor,
        gold_head: torch.Tensor,
        deprel_logits: torch.Tensor,
        gold_deprel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute head and deprel cross-entropy losses.

        Following Dozat & Manning, deprel prediction loss is conditioned on
        the gold heads.

        Standard practice is to weigh these to combine them; we return them
        separately to allow for more possibilities downstream.

        Args:
            head_logits: Head scores.
            gold_head: Gold head indices.
            deprel_logits: dependency relation scores.
            gold_deprel: Gold dependency relations.

        Returns:
            The two losses.
        """
        gold_head = self._unshift_head(gold_head)
        head_loss = self.loss_func(
            head_logits.reshape(-1, head_logits.size(-1)),
            gold_head.reshape(-1),
        )
        length = deprel_logits.size(1)
        num_deprel = deprel_logits.size(3)
        gold_head_expanded = (
            gold_head.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, length, 1, num_deprel)
        )
        # Selects the appropriate deprel logits.
        selected_deprel_logits = torch.gather(
            deprel_logits,
            dim=2,
            index=gold_head_expanded,
        ).squeeze(2)
        # TODO: consider having the caller pass in the loss function object.
        deprel_loss = self.loss_func(
            selected_deprel_logits.reshape(-1, num_deprel),
            gold_deprel.reshape(-1),
        )
        return head_loss, deprel_loss

    def decode(
        self,
        head_logits: torch.Tensor,
        deprel_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode head and deprel predictions from logits.

        This uses greedy decoding.

        Args:
            head_logits: Head scores of shape N x L x L.
            deprel_logits: Label scores of shape N x L x L x C
            mask: Attention mask of shape N x L.

        Returns:
            Predicted head and deprel.
        """
        # FIXME indices.
        pred_head = head_logits.argmax(dim=-1)
        pred_head.masked_fill_(~mask, special.PAD_IDX)
        batch_size = deprel_logits.size(0)
        length = deprel_logits.size(1)
        num_deprel = deprel_logits.size(3)
        pred_head_expanded = (
            pred_head.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(batch_size, length, 1, num_deprel)
        )
        selected_deprel_logits = torch.gather(
            deprel_logits, dim=2, index=pred_head_expanded
        )
        # FIXME indices.
        pred_deprel = selected_deprel_logits.squeeze(2).argmax(dim=-1)
        pred_head = self._shift_head(pred_head)
        pred_deprel.masked_fill_(~mask, special.PAD_IDX)
        return pred_head, pred_deprel
