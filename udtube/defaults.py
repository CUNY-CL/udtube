"""Defaults."""

from yoyodyne import optimizers, schedulers

# Scalar constants.
NEG_EPSILON = -1e7

# Default text encoding.
ENCODING = "utf-8"

# Architecture arguments.
ENCODER = "google-bert/bert-base-multilingual-cased"
ARC_MLP_SIZE = 524
LABEL_MLP_SIZE = 128
POOLING_LAYERS = 4
REVERSE_EDITS = True
USE_UPOS = True
USE_XPOS = True
USE_LEMMA = True
USE_FEATS = True
USE_PARSE = True

# Training arguments.
BATCH_SIZE = 32
DROPOUT = 0.2
OPTIMIZER = optimizers.Adam
SCHEDULER = schedulers.Dummy
