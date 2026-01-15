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
        hypo_head: torch.Tensor,
        gold_head: torch.Tensor,
        hypo_deprel: torch.Tensor | None = None,
        gold_deprel: torch.Tensor | None = None,
    ) -> None:
        """Accumulates sufficient statistics for a batch.

        Args:
            hypo_head.
            gold_head.
            hypo_deprel: required if `labeled=True`.
            gold_deprel: required if `labeled=True`.
        """
        assert hypo_head.shape == gold_head.shape, (
            f"Shape mismatch: hypo_head {hypo_head.shape} "
            f"!= gold_head {gold_head.shape}"
        )
        if self.labeled:
            assert (
                hypo_deprel is not None and gold_deprel is not None
            ), "Labels required for labeled attachment score"
            assert hypo_deprel.shape == gold_deprel.shape, (
                f"Shape mismatch: hypo_deprel {hypo_deprel.shape} "
                f"!= gold_deprel {gold_deprel.shape}"
            )
            assert hypo_deprel.shape == hypo_head.shape, (
                f"Shape mismatch: hypo_deprel {hypo_deprel.shape} "
                f"!= hypo_head {hypo_head.shape}"
            )
        mask = gold_head != self.ignore_index
        head_correct = (hypo_head == gold_head) & mask
        if self.labeled:
            # For LAS, both head and deprel must be correct.
            deprel_correct = (hypo_deprel == gold_deprel) & mask
            self.correct += torch.sum(head_correct & deprel_correct)
        else:
            # For UAS, only head needs to be correct.
            self.correct += head_correct.sum()
        self.total += mask.sum()

    def compute(self) -> torch.Tensor:
        print(self.correct.item(), self.total.item())
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
