"""CoNLL-U file parser.

This is roughly compatible with the third-party package `conllu`, though it
only has features we care about."""

from __future__ import annotations

import collections
import dataclasses
import re

from typing import Dict, Iterable, Iterator, List, Optional, TextIO, Tuple


class Error(Exception):

    pass


class ID:
    """Representation of a CoNLL-U sentence ID.

    Most of the time this is just a single integer. Two other possibilities
    exist, however:

    * decimals: represented by an integer and a second integer > 0.
    * MWEs: represented by two integers denoting a span of tokens.

    A ID cannot both have a decimal and be a MWE.

    The decimal dummy value is 0; the MWE dummy value (for the upper element
    of the span) is the same as the lower value.

    Args:
        lower (int): the lower or sole index.
        decimal (int): a decimal (> 0) on the sole index.
        upper (int, optional): the upper index, for MWEs.

    Raises:
        Error: decimal <= 0.
        Error: lower > upper.
    """

    lower: int
    decimal: int
    upper: int

    def __init__(
        self,
        lower,
        *,
        decimal: Optional[int] = None,
        upper: Optional[int] = None,
    ):
        self.lower = lower
        if decimal is None:
            self.decimal = 0
        else:
            self.decimal = int(decimal)
            if self.decimal <= 0:
                raise Error(f"decimal {decimal} <= 0")
        self.upper = self.lower if upper is None else upper
        if self.lower > self.upper:
            raise Error(f"lower {lower} > upper {upper}")

    @classmethod
    def parse_from_string(cls, string: str) -> ID:
        if mtch := re.fullmatch(r"(\d+)-(\d+)", string):
            return cls(int(mtch.group(1)), upper=int(mtch.group(2)))
        elif mtch := re.fullmatch(r"(\d+)\.(\d+)", string):
            return cls(int(mtch.group(1)), decimal=mtch.group(2))
        elif mtch := re.fullmatch(r"\d+", string):
            return cls(int(mtch.group()))
        else:
            raise Error(f"Unable to parse ID {string}")

    def __str__(self) -> str:
        if self.is_mwe:
            return f"{self.lower}-{self.upper}"
        elif self.is_decimal:
            return f"{self.lower}.{self.decimal}"
        else:
            return str(self.lower)

    def __len__(self) -> int:
        return 1 + self.upper - self.lower

    def __eq__(self, other: ID) -> bool:
        return (
            self.lower == other.lower
            and self.decimal == other.decimal
            and self.upper == other.upper
        )

    def get_slice(self) -> slice:
        if self.is_decimal:
            raise ValueError(f"cannot convert decimal ID {self!s} to slice")
        # We add one to the right end since upper bounds are open in Python.
        return slice(self.lower, self.upper + 1)

    @property
    def is_mwe(self) -> bool:
        return len(self) > 1

    @property
    def is_decimal(self) -> bool:
        return self.decimal > 0


# TODO: when dropping support for Python 3.9, add `slots=True`.
@dataclasses.dataclass
class Token:
    """Token object."""

    id_: ID  # Avoids clash with built-in.
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: str  # This could be parsed as an Optional[int] but YAGNI.
    deprel: str
    deps: str
    misc: str

    @classmethod
    def parse_from_string(cls, string: str) -> Token:
        id_, form, lemma, upos, xpos, feats, head, deprel, deps, misc = (
            string.split("\t")
        )
        return cls(
            ID.parse_from_string(id_),
            form,
            lemma,
            upos,
            xpos,
            feats,
            head,
            deprel,
            deps,
            misc,
        )

    def __str__(self) -> str:
        return (
            f"{self.id_!s}\t{self.form}\t{self.lemma}\t{self.upos}\t"
            f"{self.xpos}\t{self.feats}\t{self.head}\t{self.deprel}\t"
            f"{self.deps}\t{self.misc}"
        )

    @property
    def is_mwe(self) -> bool:
        return self.id_.is_mwe


class TokenList(collections.UserList):
    """TokenList object.

    This behaves like a list of tokens (of type Dict[str, str]) with
    optional associated metadata.

    Args:
        tokens (Iterable[Token]): iterable of tokens.
        metadata (Dict[str, Optional[str]], optional): ordered dictionary of
            string/key pairs.
    """

    metadata: Dict[str, Optional[str]]

    def __init__(self, tokens: Iterable[Token], metadata=None):
        super().__init__(tokens)
        self.metadata = metadata if metadata is not None else {}

    def __str__(self) -> str:
        line_buf = []
        for key, value in self.metadata.items():
            if value:
                line_buf.append(f"# {key} = {value}")
            else:  # `newpar` etc.
                line_buf.append(f"# {key}")
        for token in self:
            line_buf.append(str(token))
        return "\n".join(line_buf) + "\n"

    def get_tokens(self) -> List[str]:
        """List of tokens to be fed into tokenizer."""
        return [token.form for token in self if not token.is_mwe]


# Parsing.


def _maybe_parse_metadata(line: str) -> Optional[Tuple[str, str]]:
    """Attempts to parse the line as metadata."""
    # The first group is the key; the optional third element is the value.
    if match := re.fullmatch(r"#\s+(.+?)(\s+=\s+(.*))?", line):
        return match.group(1), match.group(3)


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
            tokens.append(Token.parse_from_string(line))
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
            tokens.append(Token.parse_from_string(line))
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
