"""Datasets and related utilities."""

import abc
import dataclasses
import mmap
from typing import BinaryIO, Iterator, List, Optional

import torch
from torch import nn
from torch.utils import data

from .. import defaults
from . import conllu, mappers


class Item(nn.Module):
    """Tensors representing a single labeled sentence."""

    tokenlist: conllu.TokenList
    upos: Optional[torch.Tensor]
    xpos: Optional[torch.Tensor]
    lemma: Optional[torch.Tensor]
    feats: Optional[torch.Tensor]
    head: Optional[torch.Tensor]
    label: Optional[torch.Tensor]

    def __init__(
        self,
        tokenlist,
        upos=None,
        xpos=None,
        lemma=None,
        feats=None,
        head=None,
        label=None,
    ):
        super().__init__()
        self.tokenlist = tokenlist
        self.register_buffer("upos", upos)
        self.register_buffer("xpos", xpos)
        self.register_buffer("lemma", lemma)
        self.register_buffer("feats", feats)
        self.register_buffer("head", head)
        self.register_buffer("label", label)

    def get_tokens(self) -> List[str]:
        return self.tokenlist.get_tokens()

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

    @property
    def use_head(self) -> bool:
        return self.head is not None

    @property
    def use_label(self) -> bool:
        return self.label is not None


@dataclasses.dataclass
class AbstractDataset(abc.ABC):
    """Base class for datasets.

    Args:
        path (str).
    """

    path: str


@dataclasses.dataclass
class IterableTextDataset(AbstractDataset, data.IterableDataset):
    """Iterable (non-random access), text-only dataset."""

    def __iter__(self) -> Iterator[Item]:
        for tokenlist in conllu.parse_from_path(self.path):
            yield Item(tokenlist)


@dataclasses.dataclass
class AbstractTaggedDataset(AbstractDataset):

    mapper: mappers.Mapper
    use_upos: bool
    use_xpos: bool
    use_lemma: bool
    use_feats: bool
    use_head: bool
    use_label: bool

    def tokenlist_to_item(self, tokenlist: conllu.TokenList) -> Item:
        return Item(
            tokenlist,
            upos=(
                self.mapper.encode_upos(
                    token.upos for token in tokenlist if not token.is_mwe
                )
                if self.use_upos
                else None
            ),
            xpos=(
                self.mapper.encode_xpos(
                    token.xpos for token in tokenlist if not token.is_mwe
                )
                if self.use_xpos
                else None
            ),
            lemma=(
                self.mapper.encode_lemma(
                    (token.form for token in tokenlist if not token.is_mwe),
                    (token.lemma for token in tokenlist if not token.is_mwe),
                )
                if self.use_lemma
                else None
            ),
            feats=(
                self.mapper.encode_feats(
                    token.feats for token in tokenlist if not token.is_mwe
                )
                if self.use_feats
                else None
            ),
            head=(
                (
                    self.mapper.encode_head(
                        token.head for token in tokenlist if not token.is_mwe
                    )
                    if self.use_parse
                    else None
                ),
            ),
            label=(
                (
                    self.mapper.encode_label(
                        token.label for token in tokenlist if not token.is_mwe
                    )
                    if self.use_parse
                    else None
                ),
            ),
        )


@dataclasses.dataclass
class IterableTaggedDataset(AbstractTaggedDataset, data.IterableDataset):
    """Iterable (non-random access), tagged dataset."""

    def __iter__(self) -> Iterator[Item]:
        for tokenlist in conllu.parse_from_path(self.path):
            yield self.tokenlist_to_item(tokenlist)


@dataclasses.dataclass
class MappableDataset(AbstractTaggedDataset, data.Dataset):
    """Mappable (random access), tagged dataset.

    This is implemented with a memory map after making a single pass through
    the file to compute offsets.
    """

    _offsets: List[int] = dataclasses.field(default_factory=list, init=False)
    _mmap: Optional[mmap.mmap] = dataclasses.field(default=None, init=False)
    _fobj: Optional[BinaryIO] = dataclasses.field(default=None, init=False)

    def __post_init__(self):
        # Computes offsets.
        self._offsets = []
        with open(self.path, "rb") as source:
            self._offsets.append(0)
            while line := source.readline():
                # CoNLL-U sentences are separated by blank lines.
                if line.strip():
                    continue
                offset = source.tell()
                # Confirms there is content after this blank line.
                if source.peek(1):
                    self._offsets.append(offset)

    def _get_mmap(self) -> mmap.mmap:
        # Makes this safe for use with multiple workers.
        if self._mmap is None:
            self._fobj = open(self.path, "rb")
            self._mmap = mmap.mmap(
                self._fobj.fileno(), 0, access=mmap.ACCESS_READ
            )
        return self._mmap

    def __getitem__(self, idx: int) -> Item:
        mm = self._get_mmap()
        start = self._offsets[idx]
        if idx + 1 < len(self._offsets):
            end = self._offsets[idx + 1]
        else:
            end = mm.size()
        chunk = mm[start:end].decode(defaults.ENCODING).strip()
        tokenlist = conllu.parse_from_string(chunk)
        return self.tokenlist_to_item(tokenlist)

    def __len__(self) -> int:
        return len(self._offsets)

    def __del__(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
        if self._fobj is not None:
            self._fobj.close()
