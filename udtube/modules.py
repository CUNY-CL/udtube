"""UDTube modules.

In the documentation below, N is the batch size, C is the number of classes
for a classification head, and L is the maximum length (in subwords, tokens,
or tags) of a sentence in the batch.
"""

import logging
from typing import List, Optional, Tuple

import lightning
import torch
import transformers
from torch import nn

from . import data, defaults, encoders


class UDTubeEncoder(lightning.LightningModule):
    """Encoder portion of the model.

    Args:
        dropout: Dropout probability.
        encoder: Name of the Hugging Face model used to tokenize and encode.
        pooling_layers: Number of layers to use to compute the embedding.
    """

    dropout_layer: nn.Dropout
    encoder: transformers.AutoModel
    pooling_layers: int

    def __init__(
        self,
        dropout: float = defaults.DROPOUT,
        encoder: str = defaults.ENCODER,
        pooling_layers: int = defaults.POOLING_LAYERS,
    ):
        super().__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.encoder = encoders.load(encoder, dropout)
        self.pooling_layers = pooling_layers

    # Properties.

    @property
    def hidden_size(self) -> int:
        return self.encoder.config.hidden_size

    def _group_embeddings(
        self,
        embeddings: torch.Tensor,
        tokenized: transformers.BatchEncoding,
    ) -> Tuple[torch.Tensor, List[List[str]]]:
        """Groups subword embeddings to form word embeddings.

        This is necessary because each classifier head makes per-word
        decisions, but the contextual embeddings use subwords. Therefore,
        we average over the subwords for each word.

        Args:
            embeddings: the embeddings tensor to pool.
            tokens: the batch encoding.

        Returns:
            The re-pooled embeddings tensor.
        """
        new_sentence_embeddings = []
        for sentence_encodings, sentence_embeddings in zip(
            tokenized.encodings, embeddings
        ):
            # This looks like an overly elaborate loop that could be a list
            # comprehension, but this is much faster.
            indices = []
            i = 0
            while i < len(sentence_encodings.word_ids):
                word_id = sentence_encodings.word_ids[i]
                # Have hit padding.
                if word_id is None:
                    break
                pair = sentence_encodings.word_to_tokens(word_id)
                indices.append(pair)
                # Fast-forwards to the start of the next word.
                i = pair[-1]
            # For each span of subwords, combine via mean and then stack them.
            new_sentence_embeddings.append(
                torch.stack(
                    [
                        torch.mean(sentence_embeddings[start:end], dim=0)
                        for start, end in indices
                    ]
                )
            )
        # Pads and stacks across sentences; the leading dimension is ragged
        # but `pad` cowardly refuses to pad non-trailing dimensions, so we
        # abuse transposition and permutation.
        pad_max = max(
            len(sentence_embedding)
            for sentence_embedding in new_sentence_embeddings
        )
        return torch.stack(
            [
                nn.functional.pad(
                    sentence_embedding.T,
                    (0, pad_max - len(sentence_embedding)),
                    value=0,
                )
                for sentence_embedding in new_sentence_embeddings
            ]
        ).permute(0, 2, 1)

    def forward(
        self,
        batch: data.Batch,
    ) -> torch.Tensor:
        """Computes the contextual word-level encoding.

        This truncates (if necessary), computes the subword encodings,
        stacks and mean-pools the pooling layers, applies dropout, and
        then mean-pools the subwords that make up each word.

        Args:
            batch: a data batch.

        Returns:
            A contextual word-level encoding.
        """
        # If something is longer than an allowed sequence, we trim it down.
        actual_length = batch.tokens.input_ids.shape[1]
        max_length = self.encoder.config.max_position_embeddings
        if actual_length > max_length:
            logging.warning(
                "Truncating sequence from %d to %d", actual_length, max_length
            )
            batch.tokens.input_ids = batch.tokens.input_ids[:max_length]
            batch.tokens.attention_mask = batch.tokens.attention_mask[
                :max_length
            ]
        # We move these manually rather than moving the whole batch encoding.
        x = self.encoder(
            batch.tokens.input_ids.to(self.device),
            batch.tokens.attention_mask.to(self.device),
        ).hidden_states
        # Stacks the pooling layers.
        x = torch.stack(x[-self.pooling_layers :])
        # Averages them into one embedding layer; automatically squeezes the
        # mean dimension.
        x = torch.mean(x, dim=0)
        # Applies dropout.
        x = self.dropout_layer(x)
        # Maps from subword embeddings to word-level embeddings.
        x = self._group_embeddings(x, batch.tokens)
        return x


class UDTubeClassifier(lightning.LightningModule):
    """Classifier portion of the model.

    Args:
        hidden_size: size of the encoder hidden layer.
        use_upos: enables the universal POS tagging task.
        use_xpos: enables the language-specific POS tagging task.
        use_lemma: enables the lemmatization task.
        use_feats: enables the morphological feature tagging task.
        upos_out_size: number of UPOS classes; usually set automatically.
        xpos_out_size: number of XPOS classes; usually set automatically.
        lemma_out_size: number of LEMMA classes; usually set automatically.
        feats_out_size: number of FEATS classes; usually set automatically.
    """

    upos_head: Optional[nn.Sequential]
    xpos_head: Optional[nn.Sequential]
    lemma_head: Optional[nn.Sequential]
    feats_head: Optional[nn.Sequential]

    @staticmethod
    def _make_head(hidden_size: int, out_size: int) -> nn.Sequential:
        """Helper for generating heads.

        Args:
            out_size (int).

        Returns:
            A sequential linear layer.
        """
        return nn.Sequential(
            nn.Linear(hidden_size, out_size),
            nn.LeakyReLU(),
        )

    def __init__(
        self,
        hidden_size: int,
        use_upos: bool = defaults.USE_UPOS,
        use_xpos: bool = defaults.USE_XPOS,
        use_lemma: bool = defaults.USE_LEMMA,
        use_feats: bool = defaults.USE_FEATS,
        *,
        # `2` is a dummy value here; it will be set by the dataset object.
        upos_out_size: int = 2,
        xpos_out_size: int = 2,
        lemma_out_size: int = 2,
        feats_out_size: int = 2,
        # Optimization and LR scheduling.
        **kwargs,
    ):
        super().__init__()
        self.upos_head = (
            self._make_head(hidden_size, upos_out_size) if use_upos else None
        )
        self.xpos_head = (
            self._make_head(hidden_size, xpos_out_size) if use_xpos else None
        )
        self.lemma_head = (
            self._make_head(hidden_size, lemma_out_size) if use_lemma else None
        )
        self.feats_head = (
            self._make_head(hidden_size, feats_out_size) if use_feats else None
        )

    # Properties.

    @property
    def use_upos(self) -> bool:
        return self.upos_head is not None

    @property
    def use_xpos(self) -> bool:
        return self.xpos_head is not None

    @property
    def use_lemma(self) -> bool:
        return self.lemma_head is not None

    @property
    def use_feats(self) -> bool:
        return self.feats_head is not None

    # Forward pass.

    def forward(self, encodings: torch.Tensor) -> data.Logits:
        """Computes logits for each of the classification heads.

        This takes the contextual word encodings and then computes the logits
        for each of the active classification heads.This yields logits of the
        shape N x L x C. Loss and accuracy functions expect N x C x L, so we
        permute to produce this shape.

        Args:
            encodings: the contextual word

        Returns:
            A contextual word-level encoding.
        """
        return data.Logits(
            upos=(
                self.upos_head(encodings).permute(0, 2, 1)
                if self.use_upos
                else None
            ),
            xpos=(
                self.xpos_head(encodings).permute(0, 2, 1)
                if self.use_xpos
                else None
            ),
            lemma=(
                self.lemma_head(encodings).permute(0, 2, 1)
                if self.use_lemma
                else None
            ),
            feats=(
                self.feats_head(encodings).permute(0, 2, 1)
                if self.use_feats
                else None
            ),
        )
