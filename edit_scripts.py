#!/usr/bin/env python
"""Edit script extraction and application.
This is based on:
Chrupała, G. 2014. Normalizing tweets with edit scripts and recurrent neural
embeddings. In Proceedings of the 52nd Annual Meeting of the Association for
Computational Linguistics (Volume 2: Short Papers), pages 680-686."""

__author__ = "Kyle Gorman"

import dataclasses
import difflib

from typing import Dict, List


@dataclasses.dataclass
class EditOp:
    """Representation of a single edit operation."""

    delete: bool = False
    insert: str = ""

    def __hash__(self) -> int:
        return self.delete + hash(self.insert)


class EditScript:

    _ops: List[EditOp]

    DEL = "~"
    SEP = "|"

    def __init__(self, istring: str, ostring: str):
        self._ops = []
        matcher = difflib.SequenceMatcher(a=istring, b=ostring, autojunk=False)
        table: Dict[int, EditOp] = {}
        for tag, ix, iy, ox, oy in matcher.get_opcodes():
            if tag in ("replace", "insert"):
                op = table.get(ix, EditOp())
                op.insert = ostring[ox:oy]
                table[ix] = op
            if tag in ("replace", "delete"):
                for i in range(ix, iy):
                    op = table.get(i, EditOp())
                    op.delete = True
                    table[i] = op
        self._ops = []
        if not table:
            self._ops = []
            return
        for i in range(max(table.keys()) + 1):
            self._ops.append(table.get(i, EditOp()))

    # String (de)serialization.

    def __str__(self) -> str:
        pieces = []
        for op in self._ops:
            if op.delete:
                pieces.append(self.DEL + op.insert)
            else:
                pieces.append(op.insert)
        return self.SEP.join(pieces)

    @classmethod
    def fromtag(cls, tag: str) -> "EditScript":
        script = cls.__new__(cls)
        script._ops = []
        for opstr in tag.split(cls.SEP):
            if opstr.startswith(cls.DEL):
                op = EditOp(True, opstr[1:])
            else:
                op = EditOp(False, opstr)
            script._ops.append(op)
        return script

    def apply(self, istring: str) -> str:
        pieces: List[str] = []
        for i, op in enumerate(self._ops):
            pieces.extend(op.insert)
            if not op.delete:
                pieces.append(istring[i : i + 1])
        return "".join(pieces) + istring[len(self._ops) :]


class ReverseEditScript(EditScript):
    """Same as above but works in reverse order, for "suffixal" languages."""

    @staticmethod
    def _reverse(istring: str) -> str:
        # Sadly this is 3x faster than the elegant "".join(reversed(istring)).
        return istring[::-1]

    def __init__(self, istring: str, ostring: str):
        # Reverses both strings.
        super().__init__(self._reverse(istring), self._reverse(ostring))

    def apply(self, istring: str) -> str:
        # Reverses input string and then reverse the output string.
        return self._reverse(super().apply(self._reverse(istring)))
