"""Command-line interface."""

import logging

from lightning.pytorch import callbacks as pytorch_callbacks, cli
from yoyodyne import trainers

from . import callbacks, data, models


class UDTubeCLI(cli.LightningCLI):
    """The UDTube CLI interface.

    Use with `--help` to see the full list of options.
    """

    def add_arguments_to_parser(
        self, parser: cli.LightningArgumentParser
    ) -> None:
        parser.add_lightning_class_args(
            pytorch_callbacks.ModelCheckpoint,
            "checkpoint",
        )
        parser.add_lightning_class_args(
            callbacks.PredictionWriter,
            "prediction",
        )
        # Links.
        parser.link_arguments("model.encoder", "data.encoder")
        parser.link_arguments("data.model_dir", "trainer.default_root_dir")
        parser.link_arguments(
            "data.model_dir", "trainer.logger.init_args.save_dir"
        )
        parser.link_arguments("model.reverse_edits", "data.reverse_edits")
        parser.link_arguments(
            "data.upos_tagset_size",
            "model.upos_out_size",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.xpos_tagset_size",
            "model.xpos_out_size",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.lemma_tagset_size",
            "model.lemma_out_size",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.feats_tagset_size",
            "model.feats_out_size",
            apply_on="instantiate",
        )


def main() -> None:
    logging.basicConfig(
        format="%(levelname)s: %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level="INFO",
    )
    UDTubeCLI(
        models.UDTube,
        data.DataModule,
        auto_configure_optimizers=False,
        parser_kwargs={"parser_mode": "omegaconf"},
        # Prevents prediction logits from accumulating in memory; see the
        # documentation in `trainers.py` for more context.
        trainer_class=trainers.Trainer,
    )


def python_interface(args: cli.ArgsType = None):
    """Interface to use models through Python."""
    UDTubeCLI(
        models.UDTube,
        data.DataModule,
        auto_configure_optimizers=False,
        parser_kwargs={"parser_mode": "omegaconf"},
        trainer_class=trainers.Trainer,
        args=args,
    )
