"""Special symbols."""

PAD = "<P>"
UNK = "<UNK>"
SPECIAL = [PAD, UNK]

PAD_IDX = 0
UNK_IDX = 1

# Special cases for parser.
HEAD_ROOT_IDX = 0
HEAD_PAD_IDX = -1

OFFSET = len(SPECIAL)
