"""CLI interface."""

# TODO: rename this `cli.py` once this is properly packaged.

import argparse
import logging

from pytorch_lightning import cli

import callbacks
import datamodules
import models


class UDTubeCLI(cli.LightningCLI):
    """Command-line interface for UDTube.

    For all supported options, run with `--help`.
    """

    def add_arguments_to_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--output_file",
            help="The file to output to, can be set to stdout.",
        )
        parser.link_arguments("model.path_name", "data.path_name")
        parser.link_arguments("model.path_name", "trainer.default_root_dir")
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.reverse_edits", "model.reverse_edits")
        parser.link_arguments(
            "data.pos_classes_cnt",
            "model.pos_out_label_size",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.xpos_classes_cnt",
            "model.xpos_out_label_size",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.lemma_classes_cnt",
            "model.lemma_out_label_size",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.feats_classes_cnt",
            "model.feats_out_label_size",
            apply_on="instantiate",
        )

    def before_instantiate_classes(self) -> None:
        if self.subcommand == "predict":
            self.trainer_defaults["callbacks"] = [
                callbacks.CustomWriter(self.config.predict.output_file)
            ]
        elif self.subcommand == "test":
            self.trainer_defaults["callbacks"] = [
                callbacks.CustomWriter(self.config.test.output_file)
            ]


def main() -> None:
    logging.basicConfig(
        format="%(filename)s %(levelname)s: %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level="INFO",
    )
    # The ConlluDataModule can also handle unlabeled data.
    UDTubeCLI(
        models.UDTube, datamodules.ConlluDataModule, save_config_callback=None
    )


if __name__ == "__main__":
    main()
