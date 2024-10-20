"""Defaults."""

from torch import optim

from . import schedulers

BATCH_SIZE = 32

DROPOUT = 0.5
ENCODER = "google-bert/bert-base-multilingual-cased"
POOLING_LAYERS = 4
REVERSE_EDITS = True
USE_UPOS = True
USE_XPOS = True
USE_LEMMA = True
USE_FEATS = True

# Optimization options.
OPTIMIZER = optim.Adam
SCHEDULER = schedulers.Dummy
