#!/usr/bin/env python
"""Runs a W&B sweep."""

import argparse
import functools
import logging
import sys
import tempfile
import traceback
import warnings

from typing import Any, Dict, TextIO

import torch
import yaml
from udtube import cli

import wandb

warnings.filterwarnings("ignore", ".*is a wandb run already in progress.*")


def train_sweep(config: Dict[str, Any], temp_config: TextIO) -> None:
    """Runs a single training run.

    Args:
        config: path to UDTube YAML config file.
        temp_config: temporary configuration file handle.
    """
    populate_config(config, temp_config)
    run_sweep()


def run_sweep() -> None:
    """Actually runs the sweep."""
    try:
        cli.main()
    except RuntimeError as error:
        # TODO: consider specializing this further if a solution to
        # https://github.com/pytorch/pytorch/issues/48365 is accepted.
        logging.error("Runtime error: %s", error)
    finally:
        # Clears the CUDA cache.
        torch.cuda.empty_cache()


def populate_config(config: Dict[str, Any], temp_config: TextIO) -> None:
    """Populates temporary configuration file.

    The wandb config data used here comes from the environment.

    Args:
        config: path to UDTube YAML config file.
        temp_config: temporary configuration file handle.
    """
    wandb.init()
    for key, value in wandb.config.items():
        _recursive_insert(config, key, value)
    yaml.dump(config, temp_config)


def _recursive_insert(config: Dict[str, Any], key: str, value: Any) -> None:
    """Recursively inserts values into a nested dictionary.

    Args:
        config: the config dictionary.
        key: a string with the arguments separated by ".".
        value: the value to insert.
    """
    *most, last = key.split(".")
    ptr = config
    for piece in most:
        try:
            ptr = ptr[piece]
        except KeyError:
            ptr[piece] = {}
            ptr = ptr[piece]
    if last in ptr:
        logging.debug(
            "Overriding configuration argument %s with W&B sweep value: %r",
            key,
            value,
        )
    ptr[last] = value


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r") as source:
        config = yaml.safe_load(source)
    # TODO: Consider enabling the W&B logger; we are not sure if things will
    # unless this is configured.
    temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml")
    # This makes ARGV look like an ordinary fit call so we don't have to
    # make a shell call to start UDTube.
    sys.argv = [
        sys.argv[0],
        "fit",
        "--config",
        temp_config.name,
        *sys.argv[1:],
    ]
    try:
        wandb.agent(
            sweep_id=args.sweep_id,
            entity=args.entity,
            project=args.project,
            function=functools.partial(train_sweep, config, temp_config),
            count=args.count,
        )
    except Exception:
        # Exits gracefully, so W&B logs the error.
        logging.fatal(traceback.format_exc())
        exit(1)
    finally:
        wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(filename)s %(levelname)s: %(asctime)s - %(message)s",
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
    # We pass the known args to main but remove them from ARGV.
    # See: https://docs.python.org/3/library/argparse.html#partial-parsing
    # This allows the user to override config arguments with CLI arguments.
    args, sys.argv[1:] = parser.parse_known_args()
    main(args)
