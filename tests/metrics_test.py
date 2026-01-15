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
        pred_heads = torch.tensor([[3, 3, 4, 5]])
        target_heads = torch.tensor([[3, 3, 4, 5]])
        self.metric.update(pred_heads, target_heads)
        self.assertEqual(self.metric.compute().item(), 1.0)

    def test_zero_accuracy(self):
        pred_heads = torch.tensor([[3, 4, 5, 6]])
        target_heads = torch.tensor([[4, 5, 6, 3]])
        self.metric.update(pred_heads, target_heads)
        self.assertEqual(self.metric.compute().item(), 0.0)

    def test_partial_accuracy(self):
        pred_heads = torch.tensor([[3, 3, 6, 5]])
        target_heads = torch.tensor([[3, 3, 4, 5]])
        self.metric.update(pred_heads, target_heads)
        self.assertAlmostEqual(self.metric.compute().item(), 0.75)

    def test_padding_ignored(self):
        pred_heads = torch.tensor([[3, 3, 4, special.PAD_IDX]])
        target_heads = torch.tensor([[3, 3, 4, special.PAD_IDX]])
        self.metric.update(pred_heads, target_heads)
        self.assertEqual(self.metric.compute().item(), 1.0)

    def test_padding_does_not_affect_score(self):
        pred_heads = torch.tensor([[3, 3, 4, 5]])
        target_heads = torch.tensor([[3, 3, 4, special.PAD_IDX]])
        self.metric.update(pred_heads, target_heads)
        self.assertEqual(self.metric.compute().item(), 1.0)

    def test_multiple_batches(self):
        pred_heads1 = torch.tensor([[3, 3, 4, 5]])
        target_heads1 = torch.tensor([[3, 3, 4, 5]])
        self.metric.update(pred_heads1, target_heads1)
        pred_heads2 = torch.tensor([[3, 3, 6, 5]])
        target_heads2 = torch.tensor([[3, 3, 4, 5]])
        self.metric.update(pred_heads2, target_heads2)
        self.assertAlmostEqual(self.metric.compute().item(), 0.875)

    def test_reset(self):
        pred_heads = torch.tensor([[3, 3, 4, 5]])
        target_heads = torch.tensor([[3, 3, 4, 5]])
        self.metric.update(pred_heads, target_heads)
        self.assertEqual(self.metric.compute().item(), 1.0)
        self.metric.reset()
        self.assertEqual(self.metric.compute().item(), 0.0)

    def test_empty_metric(self):
        self.assertEqual(self.metric.compute().item(), 0.0)

    def test_batch_dimension(self):
        pred_heads = torch.tensor(
            [
                [3, 3, 4, 5],
                [3, 3, 6, 5],
                [3, 3, 4, 5],
            ]
        )
        target_heads = torch.tensor(
            [
                [3, 3, 4, 5],
                [3, 3, 4, 5],
                [3, 3, 4, 5],
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
        pred_heads = torch.tensor([[3, 3, 4, 5]])
        target_heads = torch.tensor([[3, 3, 4, 5]])
        pred_labels = torch.tensor([[8, 13, 6, 10]])
        target_labels = torch.tensor([[8, 13, 6, 10]])
        self.metric.update(
            pred_heads, target_heads, pred_labels, target_labels
        )
        self.assertEqual(self.metric.compute().item(), 1.0)

    def test_zero_accuracy(self):
        pred_heads = torch.tensor([[3, 4, 5, 6]])
        target_heads = torch.tensor([[3, 6, 4, 3]])
        pred_labels = torch.tensor([[4, 5, 6, 7]])
        target_labels = torch.tensor([[8, 9, 10, 11]])
        self.metric.update(
            pred_heads, target_heads, pred_labels, target_labels
        )
        self.assertEqual(self.metric.compute().item(), 0.0)

    def test_correct_heads_wrong_labels(self):
        pred_heads = torch.tensor([[3, 3, 4, 5]])
        target_heads = torch.tensor([[3, 3, 4, 5]])
        pred_labels = torch.tensor([[4, 5, 6, 7]])
        target_labels = torch.tensor([[8, 9, 10, 11]])
        self.metric.update(
            pred_heads, target_heads, pred_labels, target_labels
        )
        self.assertEqual(self.metric.compute().item(), 0.0)

    def test_wrong_heads_correct_labels(self):
        pred_heads = torch.tensor([[3, 4, 5, 6]])
        target_heads = torch.tensor([[3, 3, 4, 5]])
        pred_labels = torch.tensor([[8, 13, 6, 10]])
        target_labels = torch.tensor([[8, 13, 6, 10]])
        self.metric.update(
            pred_heads, target_heads, pred_labels, target_labels
        )
        self.assertEqual(self.metric.compute().item(), 0.25)

    def test_partial_accuracy(self):
        pred_heads = torch.tensor([[3, 3, 6, 5]])
        target_heads = torch.tensor([[3, 3, 4, 5]])
        pred_labels = target_labels = torch.tensor([[8, 13, 6, 10]])
        self.metric.update(
            pred_heads, target_heads, pred_labels, target_labels
        )
        self.assertAlmostEqual(self.metric.compute().item(), 0.75)

    def test_padding_ignored(self):
        pred_heads = torch.tensor([[3, 3, 4, special.PAD_IDX]])
        target_heads = torch.tensor([[3, 3, 4, special.PAD_IDX]])
        pred_labels = torch.tensor([[8, 13, 6, special.PAD_IDX]])
        target_labels = torch.tensor([[8, 13, 6, special.PAD_IDX]])
        self.metric.update(
            pred_heads, target_heads, pred_labels, target_labels
        )
        self.assertEqual(self.metric.compute().item(), 1.0)

    def test_multiple_batches(self):
        pred_heads1 = target_heads1 = torch.tensor([[3, 3, 4, 5]])
        pred_labels1 = torch.tensor([[8, 13, 4, 10]])
        target_labels1 = torch.tensor([[8, 13, 6, 10]])
        self.metric.update(
            pred_heads1, target_heads1, pred_labels1, target_labels1
        )
        pred_heads2 = target_heads2 = torch.tensor([[3, 3, 6, 5]])
        pred_labels2 = torch.tensor([[8, 13, 5, 10]])
        target_labels2 = torch.tensor([[8, 13, 6, 10]])
        self.metric.update(
            pred_heads2, target_heads2, pred_labels2, target_labels2
        )
        self.assertAlmostEqual(self.metric.compute().item(), 0.75)

    def test_requires_labels(self):
        pred_heads = torch.tensor([[3, 3, 4, 5]])
        target_heads = torch.tensor([[3, 3, 4, 5]])
        with self.assertRaises(AssertionError):
            self.metric.update(pred_heads, target_heads)

    def test_reset(self):
        pred_heads = torch.tensor([[3, 3, 4, 5]])
        target_heads = torch.tensor([[3, 3, 4, 5]])
        pred_labels = torch.tensor([[8, 13, 6, 10]])
        target_labels = torch.tensor([[8, 13, 6, 10]])
        self.metric.update(
            pred_heads, target_heads, pred_labels, target_labels
        )
        self.assertEqual(self.metric.compute().item(), 1.0)
        self.metric.reset()
        self.assertEqual(self.metric.compute().item(), 0.0)

    def test_batch_dimension(self):
        pred_heads = target_heads = torch.tensor(
            [
                [3, 3, 4, 5],
                [3, 3, 4, 5],
                [3, 3, 4, 5],
            ]
        )
        pred_labels = torch.tensor(
            [
                [8, 13, 6, 10],
                [8, 13, 5, 10],
                [8, 13, 6, 10],
            ]
        )
        target_labels = torch.tensor(
            [
                [8, 13, 6, 10],
                [8, 13, 6, 10],
                [8, 13, 6, 10],
            ]
        )
        self.metric.update(
            pred_heads, target_heads, pred_labels, target_labels
        )
        self.assertAlmostEqual(self.metric.compute().item(), 11.0 / 12.0)


if __name__ == "__main__":
    unittest.main()
