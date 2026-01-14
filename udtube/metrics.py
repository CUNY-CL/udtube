"""Custom metrics.

This module implements unlabeled attachment score (UAS) and labeled attachment
score (LAS) metrics used for dependency parsing evaluation.
"""

import torch
import torchmetrics


class AttachmentScore(torchmetrics.Metric):
    """Base class for attachment scores.

    This metric computes the percentage of tokens that have the correct head
    (and optionally, correct label).
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
        hypo_labels: torch.Tensor | None = None,
        gold_labels: torch.Tensor | None = None,
    ) -> None:
        """Accumulates sufficient statistics for a batch.

        Args:
            hypo_heads.
            gold_heads.
            hypo_labels: required if `labeled=True`.
            gold_labels: required if `labeled=True`.
        """
        assert hypo_heads.shape == gold_heads.shape, (
            f"Shape mismatch: hypo_heads {hypo_heads.shape} "
            f"!= gold_heads {gold_heads.shape}"
        )
        if self.labeled:
            assert (
                hypo_labels is not None and gold_labels is not None
            ), "Labels required for labeled attachment score"
            assert hypo_labels.shape == gold_labels.shape, (
                f"Shape mismatch: hypo_labels {hypo_labels.shape} "
                f"!= gold_labels {gold_labels.shape}"
            )
            assert hypo_labels.shape == hypo_heads.shape, (
                f"Shape mismatch: hypo_labels {hypo_labels.shape} "
                f"!= hypo_heads {hypo_heads.shape}"
            )
        mask = gold_heads != self.ignore_index
        heads_correct = (hypo_heads == gold_heads) & mask
        if self.labeled:
            # For LAS, both head and label must be correct.
            labels_correct = (hypo_labels == gold_labels) & mask
            self.correct += torch.sum(heads_correct & labels_correct)
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
    ignoring the dependency label.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, labeled=False, **kwargs)


class LabeledAttachmentScore(AttachmentScore):
    """Labeled Attachment Score (LAS).

    Computes the percentage of tokens with both correct head assignment
    and correct dependency label.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, labeled=True, **kwargs)
