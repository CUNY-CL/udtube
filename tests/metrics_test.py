"""Unit tests for dependency parsing metrics."""

import unittest

import torch

from udtube import metrics, special


class UnlabeledAttachmentScoreTest(unittest.TestCase):

    def setUp(self):
        self.metric = metrics.UnlabeledAttachmentScore(
            ignore_index=special.PAD_IDX
        )

    def test_perfect_accuracy(self):
        pred_heads = torch.tensor([[0, 0, 1, 2]])
        target_heads = torch.tensor([[0, 0, 1, 2]])
        self.metric.update(pred_heads, target_heads)
        self.assertEqual(self.metric.compute().item(), 1.0)

    def test_zero_accuracy(self):
        pred_heads = torch.tensor([[0, 1, 2, 3]])
        target_heads = torch.tensor([[0, 3, 1, 0]])
        self.metric.update(pred_heads, target_heads)
        self.assertEqual(self.metric.compute().item(), 0.25)

    def test_partial_accuracy(self):
        pred_heads = torch.tensor([[0, 0, 3, 2]])
        target_heads = torch.tensor([[0, 0, 1, 2]])
        self.metric.update(pred_heads, target_heads)
        self.assertAlmostEqual(self.metric.compute().item(), 0.75)

    def test_padding_ignored(self):
        pred_heads = torch.tensor([[0, 0, 1, special.PAD_IDX]])
        target_heads = torch.tensor([[0, 0, 1, special.PAD_IDX]])
        self.metric.update(pred_heads, target_heads)
        self.assertEqual(self.metric.compute().item(), 1.0)

    def test_padding_does_not_affect_score(self):
        pred_heads = torch.tensor([[0, 0, 1, 999]])
        target_heads = torch.tensor([[0, 0, 1, special.PAD_IDX]])
        self.metric.update(pred_heads, target_heads)
        self.assertEqual(self.metric.compute().item(), 1.0)

    def test_multiple_batches(self):
        pred_heads1 = torch.tensor([[0, 0, 1, 2]])
        target_heads1 = torch.tensor([[0, 0, 1, 2]])
        self.metric.update(pred_heads1, target_heads1)
        pred_heads2 = torch.tensor([[0, 0, 3, 2]])
        target_heads2 = torch.tensor([[0, 0, 1, 2]])
        self.metric.update(pred_heads2, target_heads2)
        self.assertAlmostEqual(self.metric.compute().item(), 0.875)

    def test_reset(self):
        pred_heads = torch.tensor([[0, 0, 1, 2]])
        target_heads = torch.tensor([[0, 0, 1, 2]])
        self.metric.update(pred_heads, target_heads)
        self.assertEqual(self.metric.compute().item(), 1.0)
        self.metric.reset()
        self.assertEqual(self.metric.compute().item(), 0.0)

    def test_empty_metric(self):
        self.assertEqual(self.metric.compute().item(), 0.0)

    def test_batch_dimension(self):
        pred_heads = torch.tensor(
            [
                [0, 0, 1, 2],
                [0, 0, 3, 2],
                [0, 0, 1, 2],
            ]
        )
        target_heads = torch.tensor(
            [
                [0, 0, 1, 2],
                [0, 0, 1, 2],
                [0, 0, 1, 2],
            ]
        )
        self.metric.update(pred_heads, target_heads)
        self.assertAlmostEqual(self.metric.compute().item(), 11.0 / 12.0)


class LabeledAttachmentScoreTest(unittest.TestCase):

    def setUp(self):
        self.metric = metrics.LabeledAttachmentScore(
            ignore_index=special.PAD_IDX
        )

    def test_perfect_accuracy(self):
        pred_heads = torch.tensor([[0, 0, 1, 2]])
        target_heads = torch.tensor([[0, 0, 1, 2]])
        pred_labels = torch.tensor([[5, 10, 3, 7]])
        target_labels = torch.tensor([[5, 10, 3, 7]])
        self.metric.update(
            pred_heads, target_heads, pred_labels, target_labels
        )
        self.assertEqual(self.metric.compute().item(), 1.0)

    def test_zero_accuracy(self):
        pred_heads = torch.tensor([[0, 1, 2, 3]])
        target_heads = torch.tensor([[0, 3, 1, 0]])
        pred_labels = torch.tensor([[1, 2, 3, 4]])
        target_labels = torch.tensor([[5, 6, 7, 8]])
        self.metric.update(
            pred_heads, target_heads, pred_labels, target_labels
        )
        self.assertEqual(self.metric.compute().item(), 0.0)

    def test_correct_heads_wrong_labels(self):
        pred_heads = torch.tensor([[0, 0, 1, 2]])
        target_heads = torch.tensor([[0, 0, 1, 2]])
        pred_labels = torch.tensor([[1, 2, 3, 4]])
        target_labels = torch.tensor([[5, 6, 7, 8]])
        self.metric.update(
            pred_heads, target_heads, pred_labels, target_labels
        )
        self.assertEqual(self.metric.compute().item(), 0.0)

    def test_wrong_heads_correct_labels(self):
        pred_heads = torch.tensor([[0, 1, 2, 3]])
        target_heads = torch.tensor([[0, 0, 1, 2]])
        pred_labels = torch.tensor([[5, 10, 3, 7]])
        target_labels = torch.tensor([[5, 10, 3, 7]])
        self.metric.update(
            pred_heads, target_heads, pred_labels, target_labels
        )
        self.assertEqual(self.metric.compute().item(), 0.25)

    def test_partial_accuracy(self):
        pred_heads = torch.tensor([[0, 0, 3, 2]])
        target_heads = torch.tensor([[0, 0, 1, 2]])
        pred_labels = target_labels = torch.tensor([[5, 10, 3, 7]])
        self.metric.update(
            pred_heads, target_heads, pred_labels, target_labels
        )
        self.assertAlmostEqual(self.metric.compute().item(), 0.75)

    def test_padding_ignored(self):
        pred_heads = torch.tensor([[0, 0, 1, special.PAD_IDX]])
        target_heads = torch.tensor([[0, 0, 1, special.PAD_IDX]])
        pred_labels = torch.tensor([[5, 10, 3, special.PAD_IDX]])
        target_labels = torch.tensor([[5, 10, 3, special.PAD_IDX]])
        self.metric.update(
            pred_heads, target_heads, pred_labels, target_labels
        )
        self.assertEqual(self.metric.compute().item(), 1.0)

    def test_multiple_batches(self):
        pred_heads1 = target_heads1 = torch.tensor([[0, 0, 1, 2]])
        pred_labels1 = torch.tensor([[5, 10, 1, 7]])
        target_labels1 = torch.tensor([[5, 10, 3, 7]])
        self.metric.update(
            pred_heads1, target_heads1, pred_labels1, target_labels1
        )
        pred_heads2 = target_heads2 = torch.tensor([[0, 0, 3, 2]])
        pred_labels2 = torch.tensor([[5, 10, 2, 7]])
        target_labels2 = torch.tensor([[5, 10, 3, 7]])
        self.metric.update(
            pred_heads2, target_heads2, pred_labels2, target_labels2
        )
        self.assertAlmostEqual(self.metric.compute().item(), 0.75)

    def test_requires_labels(self):
        pred_heads = torch.tensor([[0, 0, 1, 2]])
        target_heads = torch.tensor([[0, 0, 1, 2]])
        with self.assertRaises(AssertionError):
            self.metric.update(pred_heads, target_heads)

    def test_reset(self):
        pred_heads = torch.tensor([[0, 0, 1, 2]])
        target_heads = torch.tensor([[0, 0, 1, 2]])
        pred_labels = torch.tensor([[5, 10, 3, 7]])
        target_labels = torch.tensor([[5, 10, 3, 7]])
        self.metric.update(
            pred_heads, target_heads, pred_labels, target_labels
        )
        self.assertEqual(self.metric.compute().item(), 1.0)
        self.metric.reset()
        self.assertEqual(self.metric.compute().item(), 0.0)

    def test_batch_dimension(self):
        pred_heads = target_heads = torch.tensor(
            [
                [0, 0, 1, 2],
                [0, 0, 1, 2],
                [0, 0, 1, 2],
            ]
        )
        pred_labels = torch.tensor(
            [
                [5, 10, 3, 7],
                [5, 10, 2, 7],
                [5, 10, 3, 7],
            ]
        )
        target_labels = torch.tensor(
            [
                [5, 10, 3, 7],
                [5, 10, 3, 7],
                [5, 10, 3, 7],
            ]
        )
        self.metric.update(
            pred_heads, target_heads, pred_labels, target_labels
        )
        self.assertAlmostEqual(self.metric.compute().item(), 11.0 / 12.0)


if __name__ == "__main__":
    unittest.main()
