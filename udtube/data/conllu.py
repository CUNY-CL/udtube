"""CoNLL-U file parser

This is roughly compatible with the third-party package `conllu`, though it
only has features we care about."""

import re

from typing import Dict, Iterator, List, Optional, TextIO, Tuple


# From: https://universaldependencies.org/format.html.
_fieldnames = [
    "id",
    "form",
    "lemma",
    "upos",
    "xpos",
    "feats",
    "head",
    "deprel",
    "deps",
    "misc",
]


class TokenList:
    """TokenList object.

    Args:
        tokens: List of tokens.
        metadata: ordered dictionary of string/key pairs.
    """

    tokens: List[Dict[str, str]]
    metadata: Dict[str, Optional[str]]

    def __init__(self, tokens: List[Dict[str, str]], metadata=None):
        self.tokens = tokens
        self.metadata = metadata if metadata is not None else {}

    def serialize(self) -> str:
        """Serializes the TokenList."""
        line_buf = []
        for key, value in self.metadata.items():
            if value:
                line_buf.append(f"# {key} = {value}")
            else:  # `newpar` etc.
                line_buf.append(f"# {key}")
        for token in self.tokens:
            col_buf = []
            for key in _fieldnames:
                col_buf.append(token.get(key, "_"))
            line_buf.append("\t".join(col_buf))
        return "\n".join(line_buf) + "\n"

    def __getitem__(self, index: int) -> Dict[str, str]:
        return self.tokens[index]

    def __len__(self) -> int:
        return len(self.tokens)

    def __iter__(self) -> Iterator[Dict[str, str]]:
        return iter(self.tokens)

    def __setitem__(self, index: int, value: Dict[str, str]) -> None:
        self.tokens[index] = value

    def append(self, token: Dict[str, str]) -> None:
        self.tokens.append(token)


# Parsing.


def _maybe_parse_metadata(line: str) -> Optional[Tuple[str, Optional[str]]]:
    """Attempts to parse the line as metadata."""
    # The first group is the key; the optional third element is the value.
    mtch = re.fullmatch(r"#\s+(.+?)(\s+=\s+(.*))?", line)
    if mtch:
        return mtch.group(1), mtch.group(3)


def _parse_token(line: str) -> Dict[str, str]:
    """Parses the line as a token."""
    return dict(zip(_fieldnames, line.split("\t")))


def parse_from_string(buffer: str) -> TokenList:
    """Parses a CoNLL-U sentence from a string.

    Args:
        buffer: string containing a serialized sentence.

    Return:
        TokenList.
    """
    metadata = {}
    tokens = []
    for line in buffer.splitlines():
        line = line.strip()
        maybe_metadata = _maybe_parse_metadata(line)
        if maybe_metadata:
            key, value = maybe_metadata
            metadata[key] = value
        else:
            tokens.append(_parse_token(line))
    return TokenList(tokens, metadata)


def _parse_from_handle(handle: TextIO) -> Iterator[TokenList]:
    """Incrementally parses a CoNLL-U file from an file handle.

    This does not backtrack/rewind so it can be used with streaming inputs.

    Args:
        handle: file handle opened for reading.

    Yields:
        TokenLists.
    """
    metadata = {}
    tokens = []
    for line in handle:
        line = line.strip()
        if not line:
            if tokens or metadata:
                yield TokenList(tokens.copy(), metadata.copy())
                metadata.clear()
                tokens.clear()
            continue
        maybe_metadata = _maybe_parse_metadata(line)
        if maybe_metadata:
            key, value = maybe_metadata
            metadata[key] = value
        else:
            tokens.append(_parse_token(line))
    if tokens or metadata:
        # No need to take a copy for the last one.
        yield TokenList(tokens, metadata)


def parse_from_path(path: str) -> Iterator[TokenList]:
    """Incrementally parses a CoNLL-U file from an file path.

    This does not backtrack/rewind so it can be used with streaming inputs.

    Args:
        path: path to input CoNLL-U file.

    Yields:
        TokenLists.
    """
    with open(path, "r") as source:
        yield from _parse_from_handle(source)
