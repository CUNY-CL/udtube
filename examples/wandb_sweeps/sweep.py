#!/usr/bin/env python
"""Runs a W&B sweep."""

import argparse
import functools
import logging
import subprocess
import tempfile
import traceback
import warnings

from typing import Any, Dict, List, TextIO

import wandb
import yaml

import util

warnings.filterwarnings("ignore", ".*is a wandb run already in progress.*")


def run(config: Dict[str, Any], temp_config: TextIO, argv: List[str]) -> None:
    """A single training run.

    Args:
        config: path to UDTube YAML config file.
        temp_config: temporary configuration file handle.
        argv: command-line arguments.
    """
    populate_config(config, temp_config)
    process = subprocess.Popen(argv, stderr=subprocess.PIPE, text=True)
    for line in process.stderr:
        logging.info(line.rstrip())
    wandb.finish(exit_code=process.wait())


def populate_config(
    config: Dict[str, Any], temp_config_handle: TextIO
) -> None:
    """Populates temporary configuration file.

    The wandb config data used here comes from the environment.

    Args:
        config: path to UDTube YAML config file.
        temp_config_handle: temporary configuration file handle.
    """
    wandb.init()
    for key, value in wandb.config.items():
        util.recursive_insert(config, key, value)
    yaml.dump(config, temp_config_handle)


def main(args: argparse.Namespace, argv: List[str]) -> None:
    with open(args.config, "r") as source:
        config = yaml.safe_load(source)
    # TODO: Consider enabling the W&B logger; we are not sure if things will
    # unless this is configured.
    temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml")
    argv = ["udtube", "fit", "--config", temp_config.name, *argv]
    try:
        wandb.agent(
            sweep_id=args.sweep_id,
            entity=args.entity,
            project=args.project,
            function=functools.partial(run, config, temp_config, argv),
            count=args.count,
        )
    except Exception:
        # Exits gracefully, so W&B logs the error.
        logging.fatal(traceback.format_exc())
        wandb.finish(exit_code=1)
        exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s: %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level="INFO",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_id", required=True, help="ID for the sweep.")
    parser.add_argument(
        "--entity", required=True, help="The entity scope for the project."
    )
    parser.add_argument(
        "--project", required=True, help="The project of the sweep."
    )
    parser.add_argument("--count", type=int, help="Number of runs to perform.")
    parser.add_argument("--config", required=True)
    # We separate out the args declared above, which control the sweep itself,
    # and all others, which are passed to the subprocess as is.
    args, argv = parser.parse_known_args()
    main(args, argv)
