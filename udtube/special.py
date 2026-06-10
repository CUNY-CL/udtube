"""Special symbols."""

PAD = "<P>"
UNK = "<UNK>"
SPECIAL = [PAD, UNK]

PAD_IDX = 0
UNK_IDX = 1

# Dedicated index to prevent collisions in head indices.
HEAD_PAD_IDX = -1

OFFSET = len(SPECIAL)
