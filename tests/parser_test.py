"""Tests for the biaffine dependency parser."""

import math

import torch

from udtube import parser


class TestFindCycle:
    def test_no_cycle_returns_empty(self):
        heads = [0, 0, 1, 2, 3]  # Linear chain: 1->0, 2->1, 3->2, 4->3.
        assert parser.BiaffineParser._find_cycle(heads) == []

    def test_simple_cycle(self):
        heads = [0, 2, 1, 2]  # 1->2->1.
        cycle = parser.BiaffineParser._find_cycle(heads)
        assert set(cycle) == {1, 2}

    def test_self_loop(self):
        heads = [0, 1, 0]  # Node 1 points to itself.
        assert parser.BiaffineParser._find_cycle(heads) == [1]

    def test_cycle_not_involving_root(self):
        heads = [0, 0, 3, 2, 3]  # 2->3->2.
        cycle = parser.BiaffineParser._find_cycle(heads)
        assert set(cycle) == {2, 3}


class TestChuLiuEdmonds:
    def _make_scores(self, n, arcs):
        scores = [[-math.inf] * (n + 1) for _ in range(n + 1)]
        for (h, d), s in arcs.items():
            scores[h][d] = s
        return scores

    def test_trivial_single_token(self):
        scores = self._make_scores(1, {(0, 1): 1.0})
        assert parser.BiaffineParser._chu_liu_edmonds(scores)[1] == 0

    def test_no_cycle_needed(self):
        scores = self._make_scores(
            3,
            {
                (0, 1): 5.0,
                (2, 1): 1.0,
                (0, 2): 4.0,
                (1, 2): 0.5,
                (0, 3): 1.0,
                (2, 3): 3.0,
                (1, 3): 0.5,
            },
        )
        heads = parser.BiaffineParser._chu_liu_edmonds(scores)
        assert heads[1] == 0
        assert heads[2] == 0
        assert heads[3] == 2

    def test_cycle_is_broken(self):
        # Greedy gives 1->2, 2->1 (cycle), root->1 should win entry.
        scores = self._make_scores(
            2,
            {
                (0, 1): 2.0,
                (2, 1): 3.0,
                (0, 2): 1.0,
                (1, 2): 3.0,
            },
        )
        heads = parser.BiaffineParser._chu_liu_edmonds(scores)
        assert heads[1] == 0
        assert heads[2] == 1

    def test_result_is_acyclic_on_random_scores(self):
        torch.manual_seed(0)
        n = 6
        raw = torch.randn(n + 1, n + 1).tolist()
        for i in range(n + 1):
            raw[i][i] = -math.inf
        heads = parser.BiaffineParser._chu_liu_edmonds(raw)
        assert len(heads) == n + 1
        assert parser.BiaffineParser._find_cycle(heads) == []
        for d in range(1, n + 1):
            assert 0 <= heads[d] <= n
            assert heads[d] != d

    def test_result_is_acyclic_on_many_random_seeds(self):
        """Stress-test correctness across many random inputs."""
        for seed in range(50):
            torch.manual_seed(seed)
            n = torch.randint(2, 8, ()).item()
            raw = torch.randn(n + 1, n + 1).tolist()
            for i in range(n + 1):
                raw[i][i] = -math.inf
            heads = parser.BiaffineParser._chu_liu_edmonds(raw)
            assert (
                parser.BiaffineParser._find_cycle(heads) == []
            ), f"Cycle found for seed={seed}, n={n}: {heads}"


class TestComputeLoss:
    def _make_parser(self):
        return parser.BiaffineParser(
            hidden_size=16,
            arc_mlp_size=8,
            deprel_mlp_size=8,
            num_deprel=4,
            dropout=0.0,
        )

    def test_no_negative_targets_reach_cross_entropy(self):
        my_parser = self._make_parser()
        N, L, C = 2, 5, 4
        head_logits = torch.randn(N, L, L)
        deprel_logits = torch.randn(N, L, L, C)
        gold_head = torch.tensor(
            [
                [0, 1, 0, 2, -1],
                [0, 0, 1, -1, -1],
            ]
        )
        gold_deprel = torch.tensor(
            [
                [1, 2, 3, 1, 0],
                [2, 1, 3, 0, 0],
            ]
        )
        head_loss, deprel_loss = my_parser.compute_loss(
            head_logits, gold_head, deprel_logits, gold_deprel
        )
        assert head_loss.item() > 0
        assert deprel_loss.item() > 0

    def test_deprel_pad_idx_zero_is_safe_ignore(self):
        """PAD_IDX=0 is a valid class index; the deprel loss must use
        num_deprel as ignore_index rather than 0, and must not crash."""
        my_parser = self._make_parser()
        N, L, C = 2, 4, 4
        head_logits = torch.randn(N, L, L)
        deprel_logits = torch.randn(N, L, L, C)
        gold_head = torch.tensor(
            [
                [0, 1, -1, -1],
                [0, -1, -1, -1],
            ]
        )
        # Padding positions carry PAD_IDX=0; real positions have class > 0.
        gold_deprel = torch.tensor(
            [
                [2, 3, 0, 0],
                [1, 0, 0, 0],
            ]
        )
        head_loss, deprel_loss = my_parser.compute_loss(
            head_logits, gold_head, deprel_logits, gold_deprel
        )
        assert not torch.isnan(head_loss)
        assert not torch.isnan(deprel_loss)

    def test_fully_padded_does_not_crash(self):
        my_parser = self._make_parser()
        N, L, C = 2, 3, 4
        head_logits = torch.randn(N, L, L)
        deprel_logits = torch.randn(N, L, L, C)
        gold_head = torch.full((N, L), -1)
        gold_deprel = torch.zeros(N, L, dtype=torch.long)
        head_loss, deprel_loss = my_parser.compute_loss(
            head_logits, gold_head, deprel_logits, gold_deprel
        )
        assert not torch.isnan(head_loss)
        assert not torch.isnan(deprel_loss)


class TestDecode:
    def _make_parser(self):
        return parser.BiaffineParser(
            hidden_size=16,
            arc_mlp_size=8,
            deprel_mlp_size=8,
            num_deprel=4,
            dropout=0.0,
        )

    def test_padding_positions_carry_sentinels(self):
        my_parser = self._make_parser()
        N, L, C = 2, 5, 4
        mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, False, False, False],
            ]
        )
        pred_head, pred_deprel = my_parser.decode(
            torch.randn(N, L, L),
            torch.randn(N, L, L, C),
            mask,
        )
        assert (pred_head[0, 3:] == -1).all()
        assert (pred_head[1, 2:] == -1).all()
        assert (pred_deprel[0, 3:] == 0).all()
        assert (pred_deprel[1, 2:] == 0).all()

    def test_real_positions_in_stored_range(self):
        my_parser = self._make_parser()
        N, L, C = 3, 6, 5
        torch.manual_seed(42)
        mask = torch.ones(N, L, dtype=torch.bool)
        pred_head, _ = my_parser.decode(
            torch.randn(N, L, L),
            torch.randn(N, L, L, C),
            mask,
        )
        assert (pred_head >= 0).all()
        assert (pred_head < L).all()

    def test_mst_output_is_acyclic(self):
        """Decoded MST heads must be cycle-free in 1-indexed MST space.

        _decode_sentence maps MST node 0 (virtual root) to stored 0, and MST
        node h (1-indexed) to stored h-1. The inverse is: stored 0 -> MST 0
        (root, no further traversal needed by _find_cycle since root is the
        traversal terminator); stored s > 0 -> MST s+1. We therefore check
        cycle freedom by building the 1-indexed heads list and calling
        _find_cycle, treating root-attached tokens (stored 0) as pointing to
        MST node 0.
        """
        my_parser = self._make_parser()
        for seed in range(20):
            torch.manual_seed(seed)
            N, L, C = 1, 5, 3
            mask = torch.ones(N, L, dtype=torch.bool)
            pred_head, _ = my_parser.decode(
                torch.randn(N, L, L),
                torch.randn(N, L, L, C),
                mask,
            )
            stored = pred_head[0].tolist()
            # stored s == 0 -> root (MST node 0); stored s > 0 -> MST node s+1.
            # This is lossy (both root and first-token map to stored 0) but
            # since _find_cycle stops at node 0, mapping all zeros to root is
            # the conservative correct choice for cycle detection.
            heads_1indexed = [0] + [0 if s == 0 else s + 1 for s in stored]
            assert (
                parser.BiaffineParser._find_cycle(heads_1indexed) == []
            ), f"Cycle found for seed={seed}: stored={stored}"
