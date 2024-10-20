#!/usr/bin/env python
"""Removes the "multiword" part of multi-word tokens in CoNLL-U files."""

import argparse

from udtube import data


def main(args: argparse.Namespace) -> None:
    prefix = args.source.removesuffix(".conllu")
    sink_name = prefix + "-no_mwe" + ".conllu"
    with open(sink_name, "w") as sink:
        for tokenlist in data.parse_from_path(args.source):
            tokenlist = data.TokenList(
                [token for token in tokenlist if "-" not in token["id"]],
                metadata=tokenlist.metadata,
            )
            print(tokenlist.serialize(), file=sink)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="input file")
    main(parser.parse_args())
