"""Custom metrics.

This module implements unlabeled attachment score (UAS) and labeled attachment
score (LAS) metrics used for dependency parsing evaluation.
"""

import torch
import torchmetrics


class AttachmentScore(torchmetrics.Metric):
    """Base class for attachment scores.

    This metric computes the percentage of tokens that have the correct head
    (and optionally, the correct dependency relation).
    """

    def __init__(self, labeled: bool, ignore_index: int):
        """Initialize the attachment score metric.

        Args:
            labeled: compute LAS rather than UAS?
            ignore_index: index used for padding.
        """
        super().__init__()
        self.labeled = labeled
        self.ignore_index = ignore_index
        self.add_state(
            "correct", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        hypo_heads: torch.Tensor,
        gold_heads: torch.Tensor,
        hypo_deprels: torch.Tensor | None = None,
        gold_deprels: torch.Tensor | None = None,
    ) -> None:
        """Accumulates sufficient statistics for a batch.

        Args:
            hypo_heads.
            gold_heads.
            hypo_deprels: required if `labeled=True`.
            gold_deprels: required if `labeled=True`.
        """
        assert hypo_heads.shape == gold_heads.shape, (
            f"Shape mismatch: hypo_heads {hypo_heads.shape} "
            f"!= gold_heads {gold_heads.shape}"
        )
        if self.labeled:
            assert (
                hypo_deprels is not None and gold_deprels is not None
            ), "Labels required for labeled attachment score"
            assert hypo_deprels.shape == gold_deprels.shape, (
                f"Shape mismatch: hypo_deprels {hypo_deprels.shape} "
                f"!= gold_deprels {gold_deprels.shape}"
            )
            assert hypo_deprels.shape == hypo_heads.shape, (
                f"Shape mismatch: hypo_deprels {hypo_deprels.shape} "
                f"!= hypo_heads {hypo_heads.shape}"
            )
        mask = gold_heads != self.ignore_index
        heads_correct = (hypo_heads == gold_heads) & mask
        if self.labeled:
            # For LAS, both head and deprel must be correct.
            deprels_correct = (hypo_deprels == gold_deprels) & mask
            self.correct += torch.sum(heads_correct & deprels_correct)
        else:
            # For UAS, only head needs to be correct.
            self.correct += heads_correct.sum()
        self.total += mask.sum()

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0, device=self.device)
        return self.correct.float() / self.total.float()


class UnlabeledAttachmentScore(AttachmentScore):
    """Unlabeled Attachment Score (UAS).

    Computes the percentage of tokens with correct head assignment,
    ignoring the dependency relation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, labeled=False, **kwargs)


class LabeledAttachmentScore(AttachmentScore):
    """Labeled Attachment Score (LAS).

    Computes the percentage of tokens with both correct head assignment
    and correct dependency relation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, labeled=True, **kwargs)
