"""Datasets and related utilities.

* ConlluIterDataset is a CoNLL-U dataset (labeled or not, it doedsn't matter),
  loaded incrementally.
* ConlluMapDataset is a labeled dataset, loaded greedily.
"""

import dataclasses
from typing import Iterator, List, Optional

import torch
from torch import nn
from torch.utils import data

from . import conllu, mappers
from .. import defaults


class Item(nn.Module):
    """Tensors representing a single labeled sentence."""

    tokenlist: conllu.TokenList
    upos: Optional[torch.Tensor]
    xpos: Optional[torch.Tensor]
    lemma: Optional[torch.Tensor]
    feats: Optional[torch.Tensor]

    def __init__(
        self, tokenlist, upos=None, xpos=None, lemma=None, feats=None
    ):
        super().__init__()
        self.tokenlist = tokenlist
        self.register_buffer("upos", upos)
        self.register_buffer("xpos", xpos)
        self.register_buffer("lemma", lemma)
        self.register_buffer("feats", feats)

    @property
    def tokens(self) -> List[str]:
        return [token["form"] for token in self.tokenlist]

    @property
    def use_upos(self) -> bool:
        return self.upos is not None

    @property
    def use_xpos(self) -> bool:
        return self.xpos is not None

    @property
    def use_lemma(self) -> bool:
        return self.lemma is not None

    @property
    def use_feats(self) -> bool:
        return self.feats is not None


@dataclasses.dataclass
class ConlluIterDataset(data.IterableDataset):
    """Iterable CoNLL-U data set.

    This class can be used for inference over large files because it does not
    load the whole data set into memory.

    CoNLL-U fields other than `text` are simply ignored.

    Args:
        path: path to input CoNLL-U file.
    """

    path: str

    def __iter__(self) -> Iterator[Item]:
        for tokenlist in conllu.parse_from_path(self.path):
            yield Item(tokenlist)


@dataclasses.dataclass
class ConlluMapDataset(data.Dataset):
    """Mappable CoNLL-U data set.

    This class loads the entire file into memory and is therefore only
    suitable for smaller data sets.

    It is mostly used during training and testing.
    """

    samples: List[conllu.TokenList]
    mapper: mappers.Mapper
    use_upos: bool = defaults.USE_UPOS
    use_xpos: bool = defaults.USE_XPOS
    use_lemma: bool = defaults.USE_LEMMA
    use_feats: bool = defaults.USE_FEATS

    # Required API.

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Item:
        """Retrieves item by index.

        Args:
            idx: index.

        Returns:
            Item.
        """
        tokenlist = self.samples[idx]
        return Item(
            tokenlist,
            upos=(
                self.mapper.encode_upos(token["upos"] for token in tokenlist)
                if self.use_upos
                else None
            ),
            xpos=(
                self.mapper.encode_xpos(token["xpos"] for token in tokenlist)
                if self.use_xpos
                else None
            ),
            lemma=(
                self.mapper.encode_lemma(
                    (token["form"] for token in tokenlist),
                    (token["lemma"] for token in tokenlist),
                )
                if self.use_lemma
                else None
            ),
            feats=(
                self.mapper.encode_feats(token["feats"] for token in tokenlist)
                if self.use_feats
                else None
            ),
        )
