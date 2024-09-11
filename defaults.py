import conllu

# Overriding the feats parser for the conllu reader
OVERRIDDEN_FIELD_PARSERS = {
    "feats": lambda line, i: conllu.parser.parse_nullable_value(line[i])
}
UNK_TAG = "[UNKNOWN]"
PAD_TAG = "[PAD]"
BLANK_TAG = "_"
SPECIAL_TOKENS = [UNK_TAG, PAD_TAG, BLANK_TAG]
