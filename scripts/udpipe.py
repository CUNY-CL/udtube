#!/usr/bin/env python
"""Runs UDPipe over CoNLL-U data."""

import argparse

import spacy_udpipe

from udtube import data


def main(args: argparse.Namespace) -> None:
    spacy_udpipe.download(args.langcode)
    model = spacy_udpipe.load(args.langcode)
    with open(args.output, "w") as sink:
        for tokenlist in data.parse_from_path(args.input):
            # The model insists on retokenizing the data but this is harmless.
            result = model(tokenlist.metadata["text"])
            for i, (input_token, result_token) in enumerate(
                zip(tokenlist, result)
            ):
                input_token["upos"] = result_token.pos_
                input_token["xpos"] = result_token.tag_
                input_token["lemma"] = result_token.lemma_
                input_token["feats"] = str(result_token.morph)
                tokenlist[i] = input_token
            print(tokenlist.serialize(), file=sink)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="path to input CoNLL-U file")
    parser.add_argument("output", help="path for output CoNLL-U file")
    parser.add_argument(
        "--langcode",
        required=True,
        help="the language and name of treebank (e.g., `en-ewt`); "
        "for a list of supported languages, see: "
        "https://github.com/TakeLab/spacy-udpipe/blob/master/spacy_udpipe/"
        "resources/languages.json",
    )
    main(parser.parse_args())
