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
            unique labels for label scores.
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
        self.weight = nn.Parameter(
            torch.zeros(out_size, head_size + 1, dep_size + 1)
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
    """Biaffine parser for dependency arc and label prediction.

    This takes the encoder outputs and predicts:

    * Head indices for each token.
    * Dependency labels for each arc.

    Following Dozat & Manning, we apply separate MLPs to reduce dimensionality
    before the biaffine classifiers.

    Args:
        arc_mlp_size: Hidden layer size for arc MLP.
        label_mlp_size: Hidden layer size for label MLP.
        dropout: Dropout probability for MLP layers
    """

    arc_head_mlp: nn.Module
    arc_dep_mlp: nn.Module
    arc_label_head_mlp: nn.Module
    arc_label_dep_mlp: nn.Module
    arc_attention: BiaffineAttention
    label_attention: BiaffineAttention
    loss_func: nn.CrossEntropyLoss

    def __init__(
        self,
        hidden_size,
        arc_mlp_size: int = defaults.MLP_SIZE,
        label_mlp_size: int = defaults.MLP_SIZE,
        num_labels: int = 2,  # Dummy value filled in via link.
        dropout: float = defaults.DROPOUT,
    ):
        super().__init__()
        self.arc_head_mlp = self._make_mlp(hidden_size, arc_mlp_size, dropout)
        self.arc_dep_mlp = self._make_mlp(hidden_size, arc_mlp_size, dropout)
        self.label_head_mlp = self._make_mlp(
            hidden_size, label_mlp_size, dropout
        )
        self.label_dep_mlp = self._make_mlp(
            hidden_size, label_mlp_size, dropout
        )
        self.arc_attention = BiaffineAttention(
            arc_mlp_size,
            arc_mlp_size,
            1,
        )
        self.label_attention = BiaffineAttention(
            label_mlp_size,
            label_mlp_size,
            num_labels,
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
            A (arc logits, label logits) tuple.
        """
        batch_size = encodings.size(0)
        length = encodings.size(1)
        arc_head = self.arc_head_mlp(encodings)
        arc_dep = self.arc_dep_mlp(encodings)
        label_head = self.label_head_mlp(encodings)
        label_dep = self.label_dep_mlp(encodings)
        # FIXME indices.
        arc_logits = self.arc_attention(arc_head, arc_dep).squeeze(-1)
        label_logits = self.label_attention(label_head, label_dep)
        arc_mask = mask.unsqueeze(1)
        arc_logits.masked_fill_(~arc_mask, defaults.NEG_EPSILON)
        arc_mask = arc_mask.unsqueeze(-1)
        label_logits.masked_fill_(~arc_mask, defaults.NEG_EPSILON)
        assert arc_logits.shape == (
            batch_size,
            length,
            length,
        ), f"Arc logits shape mismatch: {arc_logits.shape}"
        assert label_logits.shape == (
            batch_size,
            length,
            length,
        ), f"Label logits shape mismatch: {label_logits.shape}"
        return arc_logits, label_logits

    def compute_loss(
        self,
        gold_heads: torch.Tensor,
        head_logits: torch.Tensor,
        label_logits: torch.Tensor,
        gold_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute head and label cross-entropy losses.

        Following Dozat & Manning, label prediction loss is conditioned on
        the gold heads.

        Standard practice is to weigh these to combine
        them, but we keep them separate here to allow for more options.

        Args:
            head_logits: Head scores.
            gold_heads: Gold head indices.
            label_logits: Label scores.
            gold_labels: Gold dependency labels.

        Returns:
            Losses for head and labels.
        """
        head_loss = self.loss_func(
            head_logits.reshape(-1, head_logits.size(-1)),
            gold_heads.reshape(-1),
        )
        length = label_logits.size(1)
        num_labels = label_logits.size(2)
        gold_heads_expanded = (
            gold_heads.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, length, 1, num_labels)
        )
        # Selects the appropriate label logits.
        selected_label_logits = torch.gather(
            label_logits,
            dim=2,
            index=gold_heads_expanded,
        ).squeeze(2)
        # TODO: consider having the caller pass in the
        # loss function object instead.
        label_loss = self.loss_func(
            selected_label_logits.reshape(-1, num_labels),
            gold_labels.reshape(-1),
        )
        return head_loss, label_loss

    def decode(
        self,
        head_logits: torch.Tensor,
        label_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode head and label predictions from logits.

        This uses greedy decoding.

        Args:
            head_logits: Head scores of shape N x L x L.
            label_logits: Label scores of shape N x L x L x C
            mask: Attention mask of shape N x L.

        Returns:
            pred_heads: Predicted head indices of shape N x L.
            pred_labels: Predicted labels of shape N x L.
        """
        # FIXME indices.
        pred_heads = head_logits.argmax(dim=-1)
        pred_heads.masked_fill_(~mask, special.PAD_IDX)
        batch_size = label_logits.size(0)
        length = label_logits.size(1)
        num_labels = label_logits.size(3)
        pred_heads_expanded = (
            pred_heads.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(batch_size, length, 1, num_labels)
        )
        selected_label_logits = torch.gather(
            label_logits, dim=2, index=pred_heads_expanded
        )
        # FIXME indices.
        pred_labels = selected_label_logits.squeeze(2).argmax(dim=-1)
        pred_labels.masked_fill_(~mask, special.PAD_IDX)
        return pred_heads, pred_labels
