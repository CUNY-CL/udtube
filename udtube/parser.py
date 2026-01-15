"""Biaffine attention dependency parser.

Based on:

    Dozat, T., and Manning, C. D. 2017. Deep biaffine attention for dependency
    parsing. In ICLR.
"""

from typing import Optional, Tuple

import torch
from torch import nn

from . import defaults, special


class BiaffineAttention(nn.Module):
    r"""Biaffine attention mechanism for scoring head-dependent pairs.

    This implements the biaffine transformation:

        score(i, j) = h_j^T U h_i + (h_j \oplus h_i)^T w + b

    where h_i is the dependent representation and h_j is the head
    representation.
    """

    def __init__(
        self,
        head_size: int,
        dep_dim: int,
        out_size: int = 1,
    ):
        """Initialize biaffine attention.

        Args:
            head_size: Dimension of head representations.
            dep_size: Dimension of dependent representations.
            out_size: Output dimension; use 1 for head scores and the number of 
                labels for label scores.
        """
        super().__init__()
        self.head_dim = head_dim
        self.dep_dim = dep_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(
            torch.zeros(out_dim, head_dim + 1, dep_dim + 1, device=self.device)
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
        seq_length = head.size(1)
        assert head.shape[0] == dep.shape[0], "Batch size mismatch"
        assert head.shape[1] == dep.shape[1], "Sequence length mismatch"
        assert (
            head.shape[2] == self.head_dim
        ), f"Head dim mismatch: {head.shape[2]} != {self.head_dim}"
        assert (
            dep.shape[2] == self.dep_dim
        ), f"Dep dim mismatch: {dep.shape[2]} != {self.dep_dim}"
        # FIXME =2?
        head = torch.cat((head, torch.ones_like(head[..., :1])), dim=-1)
        # FIXME =2?
        dep = torch.cat((dep, torch.ones_like(dep[..., :1])), dim=-1)
        dep_weight = torch.einsum("bld,odh->blho", dep, self.weight)
        return torch.einsum("bsh,bdho->bdso", head, dep_weight)


class BiaffineParser(nn.Module):
    """Biaffine parser for dependency arc and label prediction.

    This module takes the encoder outputs and predicts:

    * Head indices for each token.
    * Dependency labels for each arc.

    Following Dozat & Manning, we apply separate MLPs to reduce dimensionality
    before the biaffine classifiers.
    """

    # FIXME integers
    arc_head_mlp: nn.Module
    arc_dep_mlp: nn.Module
    arc_label_head_mlp: nn.Module
    arc_label_dep_mlp: nn.Module
    arc_attention = BiaffineAttention
    label_attention = BiaffineAttention
    loss_func = nn.CrossEntropyLoss

    def __init__(
        self,
        encoder_dim: int,
        arc_mlp_dim: int,  # CONSTANTS FIXME
        label_mlp_dim: int,
        num_labels: int,
        dropout: float = defaults.DROPOUT,
    ):
        """Initialize the biaffine parser.

        Args:
            encoder_dim: Dimension of encoder output vectors
            arc_mlp_dim: Hidden dimension for arc prediction MLPs
            label_mlp_dim: Hidden dimension for label prediction MLPs
            num_labels: Number of dependency relation labels
            dropout: Dropout probability for MLP layers
        """
        super().__init__()
        self.encoder_dim = encoder_dim
        self.arc_mlp_dim = arc_mlp_dim
        self.label_mlp_dim = label_mlp_dim
        self.num_labels = num_labels
        self.arc_head_mlp = self._build_mlp(encoder_dim, arc_mlp_dim, dropout)
        self.arc_dep_mlp = self._build_mlp(encoder_dim, arc_mlp_dim, dropout)
        self.label_head_mlp = self._build_mlp(
            encoder_dim, label_mlp_dim, dropout
        )
        self.label_dep_mlp = self._build_mlp(
            encoder_dim, label_mlp_dim, dropout
        )
        self.arc_attention = BiaffineAttention(
            head_dim=arc_mlp_dim,
            dep_dim=arc_mlp_dim,
            out_dim=1,
        )
        self.label_attention = BiaffineAttention(
            head_dim=label_mlp_dim,
            dep_dim=label_mlp_dim,
            out_dim=num_labels,
        )
        self.loss_func = nn.CrossEntropyLoss(ignore_index=special.PAD_IDX)

    @staticmethod
    def _build_mlp(
        input_dim: int, hidden_dim: int, dropout: float
    ) -> nn.Module:
        """Build a single-layer MLP with ReLU activation and dropout.

        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden/output dimension.
            dropout: Dropout probability.

        Returns:
            A sequential MLP module.
        """
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        encodings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for dependency parsing.

        Args:
            encodings: Encoder output.
            mask: Optional attention mask of shape B x L.

        Returns:
            A (head logits, label logits) tuple.
        """
        # FIXME upack this more nicely.
        batch_size, seq_len, enc_dim = encodings.shape
        assert (
            enc_dim == self.encoder_dim
        ), f"Encoder dim mismatch: {enc_dim} vs {self.encoder_dim}"
        arc_head = self.arc_head_mlp(encodings)
        arc_dep = self.arc_dep_mlp(encodings)
        label_head = self.label_head_mlp(encodings)
        label_dep = self.label_dep_mlp(encodings)
        # FIXME indices
        arc_scores = self.arc_attention(arc_head, arc_dep).squeeze(-1)
        label_logits = self.label_attention(label_head, label_dep)
        # FIXME do I ever pass a mask?
        if mask is not None:
            head_mask = mask.unsqueeze(1)
            arc_logits = arc_logits.masked_fill(
                ~head_mask, defaults.NEG_EPSILON
            )
            # FIXME indices.
            head_mask_4d = head_mask.unsqueeze(-1)
            label_logits = label_logits.masked_fill(
                ~head_mask_4d, defaults.NEG_EPSILON
            )
        assert arc_logits.shape == (
            batch_size,
            seq_len,
            seq_len,
        ), f"Arc logits shape mismatch: {arc_logits.shape}"
        assert label_logits.shape == (
            batch_size,
            seq_len,
            seq_len,
            self.num_labels,
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

        Args:
            head_logits: Head scores.
            label_logits: Label scores.
            target_heads: Gold head indices.
            target_labels: Gold dependency labels.

        Returns:
            Losses for head and labels.
        """
        head_loss = self.loss_func(
            head_logits.reshape(-1, head_logits.size(-1)),
            target_heads.reshape(-1),
        )
        # FIXME nicer unpacking.
        batch_size, seq_len, _, num_labels = label_logits.shape
        target_heads_expanded = (
            target_heads.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(batch_size, seq_len, 1, num_labels)
        )
        selected_label_logits = torch.gather(
            label_logits, dim=2, index=target_heads_expanded
        ).squeeze(2)
        label_loss = self.loss_func(
            selected_label_logits.reshape(-1, num_labels),
            target_labels.reshape(-1),
        )
        return head_loss, label_loss

    def decode(
        self,
        head_logits: torch.Tensor,
        label_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode head and label predictions from logits.

        This uses greedy decoding.

        Args:
            head_logits: Head scores of shape B x L x L.
            label_logits: Label scores of shape B x L x L x C
            mask: Optional attention mask of shape B x L.

        Returns:
            pred_heads: Predicted head indices of shape B x L.
            pred_labels: Predicted labels of shape B x L.
        """
        # FIXME indices.
        pred_heads = head_logits.argmax(dim=-1)
        # FIXME unpacking.
        batch_size, seq_len, _, num_labels = label_logits.shape
        # FIXME unpacking.
        pred_heads_expanded = pred_heads.unsqueeze(-1).unsqueeze(-1).expand(batch_size, seq_len, 1, num_labels)
        selected_label_logits = torch.gather(
            label_logits, dim=2, index=pred_heads_expanded
        )
        # FIXME indices.
        pred_labels = selected_label_logits.squeeze(2).argmax(dim=-1)
        # FIXME is this ever absent?
        if mask is not None:
            pred_heads = pred_heads.masked_fill(~mask, special.PAD_IDX)
            pred_labels = pred_labels.masked_fill(~mask, special.PAD_IDX)
        return pred_heads, pred_labels
