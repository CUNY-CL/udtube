"""Collators."""

import dataclasses
import logging
from typing import Any, Iterable, List

import torch
from torch import nn
import transformers

from .. import special
from . import batches, datasets


class Error(Exception):
    pass


@dataclasses.dataclass
class Collator:
    """Collator for CoNLL-U data."""

    tokenizer: transformers.AutoTokenizer

    def __call__(self, itemlist: List[datasets.Item]) -> batches.Batch:
        # Runs the tokenizer.
        tokenized = self.tokenizer(
            [item.get_tokens() for item in itemlist],
            padding="longest",
            truncation=False,
            return_tensors="pt",
            is_split_into_words=True,
            add_special_tokens=False,
        )
        actual_length = tokenized.input_ids.size(1)
        max_length = self.tokenizer.model_max_length
        if actual_length > max_length:
            # By construction, a sequence is too long if the first element
            # beyond the encoder's max length is not padding.
            keep_items = tokenized.input_ids[:, max_length] == special.PAD_IDX
            # Ideally we'd just shorten the tag tensors, but mapping subword
            # tokens to tags is sufficiently complex that we don't know at
            # this stage how many tags to keep or get rid of. Therefore we
            # just discard these sentences.
            logging.warning(
                "Discarding %d sequence(s) exceeding the encoder's "
                "maximum length (%d)",
                torch.sum(~keep_items),
                max_length,
            )
            itemlist = self._keep_list(itemlist, keep_items)
            # In the very unlikely case that every sequence in the batch
            # exceeds the length, we raise an error, since none of the
            # subsequent steps can handle an empty batch.
            if not itemlist:
                raise Error(
                    "Every sequence in the batch exceeds the "
                    f"encoder's maximum length {max_length}"
                )
            # Not all of these have setters, so we store pointers instead.
            input_ids = tokenized.input_ids[keep_items, :max_length]
            attention_mask = tokenized.attention_mask[keep_items, :max_length]
            encodings = self._keep_list(tokenized.encodings, keep_items)
        else:
            # Grabs pointers.
            input_ids = tokenized.input_ids
            attention_mask = tokenized.attention_mask
            encodings = tokenized.encodings
        return batches.Batch(
            tokenlists=[item.tokenlist for item in itemlist],
            input_ids=input_ids,
            attention_mask=attention_mask,
            encodings=encodings,
            # Pads and stacks data for whichever classification tasks are
            # enabled.
            upos=(
                self.pad_tensors([item.upos for item in itemlist])
                if itemlist[0].use_upos
                else None
            ),
            xpos=(
                self.pad_tensors([item.xpos for item in itemlist])
                if itemlist[0].use_xpos
                else None
            ),
            lemma=(
                self.pad_tensors([item.lemma for item in itemlist])
                if itemlist[0].use_lemma
                else None
            ),
            feats=(
                self.pad_tensors([item.feats for item in itemlist])
                if itemlist[0].use_feats
                else None
            ),
            head=(
                (
                    self.pad_tensors([item.head for item in itemlist])
                    if itemlist[0].use_head
                    else None
                ),
            ),
            label=(
                (
                    self.pad_tensors([item.label for item in itemlist])
                    if itemlist[0].use_label
                    else None
                ),
            ),
        )

    @staticmethod
    def _keep_list(items: List[Any], keep_items: Iterable[bool]) -> List[Any]:
        """Simulates items[keep_items] for lists.

        Args:
            items: a list.
            keep_items: an iterable where True indicates
                that the corresponding element in `items` should be
                preserved in the output.

        Returns:
            A filtered list.
        """
        return [item for item, keep in zip(items, keep_items) if keep]

    @staticmethod
    def pad_tensors(
        tensorlist: List[torch.Tensor],
    ) -> torch.Tensor:
        """Pads and stacks a list of tensors.

        Args:
            tensorlist: a list of tensors to be padded.

        Returns:
            The padded and stacked tensor.
        """
        pad_max = max(len(tensor) for tensor in tensorlist)
        return torch.stack(
            [
                nn.functional.pad(
                    tensor, (0, pad_max - len(tensor)), value=special.PAD_IDX
                )
                for tensor in tensorlist
            ]
        )
