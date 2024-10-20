#!/usr/bin/env python
"""UDTube evaluation."""

import argparse
import logging

from udtube import data


class Error(Exception):
    pass


def main(args: argparse.Namespace) -> None:
    correct_upos = 0
    correct_xpos = 0
    correct_lemma = 0
    correct_feats = 0
    total = 0
    for linenum, (gold_list, hypo_list) in enumerate(
        zip(data.parse_from_path(args.gold), data.parse_from_path(args.hypo)),
        1,
    ):
        if len(gold_list) != len(hypo_list):
            raise Error("Length mismatch on sentence %d", linenum)
        for gold_tok, hypo_tok in zip(gold_list, hypo_list):
            correct_upos += gold_tok["upos"] == hypo_tok["upos"]
            correct_xpos += gold_tok["xpos"] == hypo_tok["xpos"]
            # We do this case-insensitively.
            correct_lemma += (
                gold_tok["lemma"].casefold() == hypo_tok["lemma"].casefold()
            )
            correct_feats += gold_tok["feats"] == hypo_tok["feats"]
            total += 1
    # Expresses this as a percentage as per the community preference.
    logging.info("UPOS accuracy:\t%2.2f", 100 * correct_upos / total)
    logging.info("XPOS accuracy:\t%2.2f", 100 * correct_xpos / total)
    logging.info("LEMMA accuracy:\t%2.2f", 100 * correct_lemma / total)
    logging.info("FEATS accuracy:\t%2.2f", 100 * correct_feats / total)


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level="INFO")
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", help="path to gold .conllu file")
    parser.add_argument("hypo", help="path to hypo .conllu file")
    main(parser.parse_args())
