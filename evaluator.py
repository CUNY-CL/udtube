"""An very simple evaluation script to confirm the performance of a model"""

import conllu
import argparse

from defaults import OVERRIDDEN_FIELD_PARSERS


def set_up_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "gold_file",
        help="the file which contains the correct annotations in conllu format",
    )
    parser.add_argument(
        "pred_file",
        help="the file which contains the predicted annotations in conllu format",
    )
    return parser


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    with open(args.gold_file) as gf:
        gold_data = conllu.parse_incr(
            gf, field_parsers=OVERRIDDEN_FIELD_PARSERS
        )
        with open(args.pred_file) as pf:
            pred_data = conllu.parse_incr(
                pf, field_parsers=OVERRIDDEN_FIELD_PARSERS
            )
            correct_pos = 0
            correct_xpos = 0
            correct_lemma = 0
            correct_feats = 0
            total = 0
            for gold_list, pred_list in zip(gold_data, pred_data):
                gold_idx = 0
                pred_idx = 0
                for _ in zip(gold_list, pred_list):
                    try:
                        gold_tok = gold_list[gold_idx]
                        pred_tok = pred_list[pred_idx]
                    except:
                        logger.warning(
                            "Sequence mismatch for:",
                            gold_list,
                            "\n",
                            pred_list,
                        )
                        total += 1
                        continue
                    if (
                        gold_tok["form"] != pred_tok["form"]
                        and "[UNK]" not in pred_tok["form"]
                    ):
                        # trying to fix alignment, but only looking within +/- 1 (only if not an [UNK] word)
                        if (
                            len(pred_list) > pred_idx + 1
                            and gold_tok["form"]
                            == pred_list[pred_idx + 1]["form"]
                        ):
                            pred_idx += 1
                            pred_tok = pred_list[pred_idx]
                        if (
                            len(gold_list) > gold_idx + 1
                            and gold_list[gold_idx + 1] == pred_tok["form"]
                        ):
                            gold_idx += 1
                            gold_tok = gold_list[gold_idx]
                    if isinstance(gold_tok["id"], tuple) and isinstance(
                        pred_tok["id"], tuple
                    ):
                        # not checking acc of multiword tokens, since they arent' annotated
                        continue
                    correct_pos += gold_tok["upos"] == pred_tok["upos"]
                    correct_xpos += gold_tok["xpos"] == pred_tok["xpos"]
                    correct_lemma += (
                        gold_tok["lemma"].casefold()
                        == pred_tok["lemma"].casefold()
                    )
                    correct_feats += gold_tok["feats"] == pred_tok["feats"]
                    total += 1
                    gold_idx += 1
                    pred_idx += 1
            print("POS ACC:", correct_pos / total)
            print("XPOS ACC:", correct_xpos / total)
            print("LEMMA ACC:", correct_lemma / total)
            print("UFEATS ACC:", correct_feats / total)
