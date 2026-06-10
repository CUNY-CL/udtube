"""Biaffine attention dependency parser.

Based on:

    Dozat, T., and Manning, C. D. 2017. Deep biaffine attention for dependency
    parsing. In ICLR.

MST decoding uses the Chu-Liu/Edmonds algorithm for maximum spanning
arborescences:

    Chu, Y.-J., and Liu, T.-H. 1965. On the shortest arborescence of a
    directed graph. Science Sinica 14:1396-1400.

    Edmonds, J. 1967. Optimum branchings. Journal of Research of the National
    Bureau of Standards 71B:233-240.

CoNLL-U head indices are 1-based for real tokens (token k has CoNLL-U id k)
and 0 for the abstract root. Arc logits have shape N x L x L, giving L
candidate head positions (0-indexed). To fit L+1 possible CoNLL-U values
(0..L) into L positions we use position 0 as a proxy for the abstract root:

    encode: stored = max(0, conllu - 1)  [root: 0->0; real token k: k->k-1]
    decode: conllu = stored + 1          [but see _decode_sentence for root]

Root-attached tokens and first-token-as-head tokens both map to stored 0,
which is a minor training ambiguity accepted by all standard implementations
that lack an explicit ROOT token in the encoder. The MST algorithm resolves
root attachment structurally at decode time.

HEAD_PAD_IDX (-1) is the unambiguous padding sentinel and is never a valid
stored head value.
"""

import math

import torch
from torch import nn

from . import defaults, special


class BiaffineAttention(nn.Module):
    """Biaffine attention mechanism for scoring head-dependent pairs.

    Implements the transformation:

        score(i, j) = h_j^T U h_i + (h_j \\oplus h_i)^T w + b

    where h_i is the dependent representation and h_j is the head
    representation.

    Args:
        head_size: Size of head representation.
        dep_size: Size of dependent representation.
        out_size: Output dimension; 1 for arc scores, num_deprel for label
            scores.
    """

    head_size: int
    dep_size: int
    out_size: int
    weight: nn.Parameter

    def __init__(self, head_size: int, dep_size: int, out_size: int = 1):
        super().__init__()
        self.head_size = head_size
        self.dep_size = dep_size
        self.out_size = out_size
        self.weight = nn.Parameter(
            torch.zeros(out_size, head_size + 1, dep_size + 1)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, head: torch.Tensor, dep: torch.Tensor) -> torch.Tensor:
        """Computes biaffine attention scores.

        Args:
            head: Head representations of shape N x L x head_size.
            dep: Dependent representations of shape N x L x dep_size.

        Returns:
            Score tensor of shape N x L x L x out_size.
        """
        assert head.shape[0] == dep.shape[0], "Batch size mismatch"
        assert head.shape[1] == dep.shape[1], "Sequence length mismatch"
        assert (
            head.shape[2] == self.head_size
        ), f"Head size mismatch: {head.shape[2]} != {self.head_size}"
        assert (
            dep.shape[2] == self.dep_size
        ), f"Dep size mismatch: {dep.shape[2]} != {self.dep_size}"
        head = torch.cat((head, torch.ones_like(head[..., :1])), dim=2)
        dep = torch.cat((dep, torch.ones_like(dep[..., :1])), dim=2)
        dep_weight = torch.einsum("bld,odh->blho", dep, self.weight)
        return torch.einsum("bsh,bdho->bdso", head, dep_weight)


class BiaffineParser(nn.Module):
    """Biaffine parser for dependency arc and label prediction.

    See module docstring for the head index representation.

    Args:
        hidden_size: Encoder hidden size.
        arc_mlp_size: Hidden size for arc scoring MLPs.
        deprel_mlp_size: Hidden size for label scoring MLPs.
        num_deprel: Number of dependency relation classes.
        dropout: Dropout probability.
    """

    arc_head_mlp: nn.Module
    arc_dep_mlp: nn.Module
    deprel_head_mlp: nn.Module
    deprel_dep_mlp: nn.Module
    arc_attention: BiaffineAttention
    deprel_attention: BiaffineAttention

    def __init__(
        self,
        hidden_size: int,
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
        self.arc_attention = BiaffineAttention(arc_mlp_size, arc_mlp_size, 1)
        self.deprel_attention = BiaffineAttention(
            deprel_mlp_size, deprel_mlp_size, num_deprel
        )

    @staticmethod
    def _make_mlp(
        input_size: int, hidden_size: int, dropout: float
    ) -> nn.Module:
        """Builds a single-layer MLP with ReLU activation and dropout."""
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        encodings: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for dependency parsing.

        Args:
            encodings: Encoder output of shape N x L x H.
            mask: Boolean word-level mask of shape N x L.

        Returns:
            arc_logits of shape N x L x L and deprel_logits of shape
            N x L x L x num_deprel. arc_logits[n, d, h] scores arc h->d.
        """
        batch_size = encodings.size(0)
        length = encodings.size(1)
        arc_logits = self.arc_attention(
            self.arc_head_mlp(encodings), self.arc_dep_mlp(encodings)
        ).squeeze(3)
        deprel_logits = self.deprel_attention(
            self.deprel_head_mlp(encodings), self.deprel_dep_mlp(encodings)
        )
        # Masks padding columns (candidate heads) so they are never selected.
        # arc_mask is N x 1 x L and broadcasts over the dependent dimension.
        arc_mask = mask.unsqueeze(1)
        arc_logits.masked_fill_(~arc_mask, defaults.NEG_EPSILON)
        deprel_logits.masked_fill_(
            ~arc_mask.unsqueeze(3), defaults.NEG_EPSILON
        )
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes arc and label cross-entropy losses.

        Following Dozat & Manning, the label loss is conditioned on gold heads.
        Both losses are returned separately so the caller can weight them.

        Args:
            head_logits: Arc scores of shape N x L x L.
            gold_head: Stored head indices of shape N x L (HEAD_PAD_IDX for
                padding; see module docstring for the encoding).
            deprel_logits: Label scores of shape N x L x L x C.
            gold_deprel: Gold label indices of shape N x L (PAD_IDX for
                padding).

        Returns:
            The head and deprel losses.
        """
        # head_logits reshaped to (N*L) x L; gold_head reshaped to (N*L,).
        # Stored values are in [0, L-1] for real tokens; HEAD_PAD_IDX=-1 is
        # the ignore_index.
        head_loss = nn.functional.cross_entropy(
            head_logits.reshape(-1, head_logits.size(2)),
            gold_head.reshape(-1),
            ignore_index=special.HEAD_PAD_IDX,
        )
        length = deprel_logits.size(1)
        num_deprel = deprel_logits.size(3)
        # Padding positions have HEAD_PAD_IDX=-1 which is out of bounds for
        # gather; clamps to 0 (the gathered values are masked out by the deprel
        # loss's ignore_index anyway).
        safe_gold_head = gold_head.clamp(min=0)
        gold_head_expanded = (
            safe_gold_head.unsqueeze(2)
            .unsqueeze(3)
            .expand(-1, length, 1, num_deprel)
        )
        selected_deprel_logits = torch.gather(
            deprel_logits, dim=2, index=gold_head_expanded
        ).squeeze(2)
        deprel_loss = nn.functional.cross_entropy(
            selected_deprel_logits.reshape(-1, num_deprel),
            gold_deprel.reshape(-1),
            ignore_index=special.PAD_IDX,
        )
        return head_loss, deprel_loss

    @staticmethod
    def _find_cycle(heads: list[int]) -> list[int]:
        """Finds a cycle in a head list, if one exists.

        Args:
            heads: heads[i] is the head of node i; heads[0] is unused (root).

        Returns:
            A list of node indices forming a cycle, or an empty list if no
                cycle is found.
        """
        n = len(heads) - 1
        visited = [False] * (n + 1)
        on_stack = [False] * (n + 1)
        for start in range(1, n + 1):
            if visited[start]:
                continue
            path: list[int] = []
            node = start
            while node != 0 and not visited[node]:
                if on_stack[node]:
                    return path[path.index(node) :]
                on_stack[node] = True
                path.append(node)
                node = heads[node]
            for p in path:
                visited[p] = True
                on_stack[p] = False
        return []

    @staticmethod
    def _chuliu_edmonds(scores: list[list[float]]) -> list[int]:
        """Maximum spanning arborescence via Chu-Liu/Edmonds.

        Node 0 is the virtual root and has no incoming arc.

        Args:
            scores: (n+1) x (n+1) dense score matrix; scores[h][d] is the
                score of arc h->d. Diagonal entries are ignored.

        Returns:
            heads where heads[d] is the predicted head of node d for
            d in 1..n; heads[0] = 0 (unused).
        """
        n = len(scores) - 1
        heads = [0] + [
            max(
                (h for h in range(n + 1) if h != d),
                key=lambda h: scores[h][d],
            )
            for d in range(1, n + 1)
        ]
        cycle = BiaffineParser._find_cycle(heads)
        if cycle:
            return heads
        cycle_set = set(cycle)
        cycle_score = {c: scores[heads[c]][c] for c in cycle}
        # Builds contracted graph: collapses cycle nodes into a super-node.
        # Non-cycle nodes are renumbered contiguously; the super-node is last.
        remap: list[int] = []
        counter = 0
        old_to_new = {}
        for node in range(n + 1):
            if node not in cycle_set:
                old_to_new[node] = counter
                remap.append(node)
                counter += 1
        super_idx = counter
        for node in cycle:
            old_to_new[node] = super_idx
        new_n = super_idx
        new_scores: list[list[float]] = [
            [-math.inf] * (new_n + 1) for _ in range(new_n + 1)
        ]
        # best_entry tracks which old (h, d) pair produced the best adjusted
        # score for arcs entering the super-node, needed for cycle resolution.
        best_entry = {}  # (nh, super_idx) -> (old_h, old_d)
        for h in range(n + 1):
            for d in range(1, n + 1):
                if h == d:
                    continue
                nh, nd = old_to_new[h], old_to_new[d]
                if nh == nd:
                    continue
                adj = scores[h][d] - (cycle_score[d] if nd == super_idx else 0)
                if adj > new_scores[nh][nd]:
                    new_scores[nh][nd] = adj
                    if nd == super_idx:
                        best_entry[(nh, super_idx)] = (h, d)
        new_heads = BiaffineParser._chuliu_edmonds(new_scores)
        result = [0] + [
            remap[new_heads[old_to_new[d]]] if d not in cycle_set else heads[d]
            for d in range(1, n + 1)
        ]
        super_head_new = new_heads[super_idx]
        old_h, best_d = best_entry[(super_head_new, super_idx)]
        result[best_d] = old_h
        return result

    def _decode_sentence(
        self, arc_scores: torch.Tensor, length: int
    ) -> torch.Tensor:
        """Decodes a single sentence via Chu-Liu/Edmonds.

        Args:
            arc_scores: Score tensor of shape L x L (entry [d, h] = score of
                arc h->d).
            length: Number of real tokens.

        Returns:
            Stored head indices of shape L (HEAD_PAD_IDX at padding positions).
        """
        # MST graph: node 0 = virtual root, nodes 1..length = real tokens.
        # scores[h][d] = score of arc h->d in 1-indexed space.
        # Root scores: score(root->d) = arc_scores[d-1, 0] (position 0 as
        # root proxy, see module docstring).
        # Real-token arc scores: arc_scores[d-1, h-1] for h,d in 1..length.
        scores = [[-math.inf] * (length + 1) for _ in range(length + 1)]
        for d in range(1, length + 1):
            scores[0][d] = arc_scores[d - 1, 0].item()
            for h in range(1, length + 1):
                if h != d:
                    scores[h][d] = arc_scores[d - 1, h - 1].item()
        mst_heads = self._chuliu_edmonds(scores)
        # Converts MST 1-indexed heads back to stored representation.
        # Virtual root (mst_heads[d] == 0) -> stored 0 (root proxy).
        # Real token head h (1-indexed) -> stored h-1 (0-indexed position).
        result = torch.full(
            (arc_scores.size(0),), special.HEAD_PAD_IDX, dtype=torch.long
        )
        for d in range(1, length + 1):
            h = mst_heads[d]
            result[d - 1] = 0 if h == 0 else h - 1
        return result

    def decode(
        self,
        head_logits: torch.Tensor,
        deprel_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decodes head and deprel predictions via MST.

        Args:
            head_logits: Arc scores of shape N x L x L.
            deprel_logits: Label scores of shape N x L x L x C.
            mask: Word-level mask of shape N x L.

        Returns:
            Stored head indices of shape N x L (HEAD_PAD_IDX at padding), and
            predicted deprel indices of shape N x L (PAD_IDX at padding).
        """
        batch_size = head_logits.size(0)
        length = head_logits.size(1)
        num_deprel = deprel_logits.size(3)
        pred_head = torch.full(
            (batch_size, length),
            special.HEAD_PAD_IDX,
            dtype=torch.long,
            device=head_logits.device,
        )
        for i, sent_len in enumerate(mask.sum(dim=1).tolist()):
            pred_head[i] = self._decode_sentence(head_logits[i], int(sent_len))
        # Gathers label logits at each predicted head position.
        # HEAD_PAD_IDX=-1 is out of bounds for gather; clamps to 0.
        pred_head_expanded = (
            pred_head.clamp(min=0)
            .unsqueeze(2)
            .unsqueeze(3)
            .expand(batch_size, length, 1, num_deprel)
        )
        pred_deprel = (
            torch.gather(deprel_logits, dim=2, index=pred_head_expanded)
            .squeeze(2)
            .argmax(dim=2)
        )
        pred_deprel.masked_fill_(~mask, special.PAD_IDX)
        return pred_head, pred_deprel
