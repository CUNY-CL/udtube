import difflib
import glob
import logging
import os
import re
import shutil

from udtube_package.UDTube.cli import udtube_python_interface

BASELINE_DIR = "datasets/baseline_predictions"
PREDICTIONS_DIR = "datasets/current_predictions"
NUM_BASELINES = 3


class MisconfigurationError(Exception):
    pass


class TestFailedError(Exception):
    pass


def pretest_validations(configs_path: str) -> None:
    """Validate if we can run tests on it here. The goal is to try to catch any silly directory management errors

    Args:
        configs_path: file path to where the configuration files can be found
    """
    if not os.path.exists(configs_path) or not os.listdir(configs_path):
        raise MisconfigurationError(
            f"The configs directory {configs_path} must not be empty. Was something deleted?"
        )
    if not os.path.exists(BASELINE_DIR) or not os.listdir(BASELINE_DIR):
        raise MisconfigurationError(
            f"The baseline directory {BASELINE_DIR} must not be empty. Was something deleted?"
        )
    if len(os.listdir(BASELINE_DIR)) != NUM_BASELINES:
        raise MisconfigurationError(
            f"The baseline directory {BASELINE_DIR} contains a different number of files than "
            f"expected. {NUM_BASELINES} is the expected amount"
        )


def _run_model(yaml_config: str, prediction_path: str) -> None:
    try:
        udtube_python_interface(["fit", f"--config={yaml_config}", f"--prediction.path={prediction_path}"])
        udtube_python_interface(
            [
                "predict",
                f"--config={yaml_config}",
                f"--prediction.path={prediction_path}",
            ]
        )
    except Exception as error:
        raise error
    finally:
        # clean tmp path, but leave empty folder for convenience
        shutil.rmtree("records/tmp/")


def run_models(configs_path: str, prediction_dir: str, suffix: str) -> None:
    """Train and predict using several configs

    Args:
        configs_path: path to where the config.yaml files that correspond to baselines are
        prediction_dir: path to where the predictions will be written to
        suffix: the full name of the file, excluding filepath
    Returns:
        N/A writes to prediction dir
    """
    for yaml_config in glob.iglob(f"{configs_path}/*"):
        # minor str processing to generate prediction file names
        name_prefix_match = re.match(r".+_", os.path.basename(yaml_config))
        if name_prefix_match:
            name_prefix = name_prefix_match.group(0)
        else:
            raise MisconfigurationError(f"No file in target dir with expected prefix")
        pred_file_name = f"{prediction_dir}/{name_prefix}{suffix}"
        logging.info(
            f"running {yaml_config} and recording test results to {pred_file_name}"
        )
        _run_model(yaml_config, pred_file_name)


def reset_baselines() -> None:
    """Reset the baseline to compare tests to.

    Should only be done in rare circumstances when we expect a change in results going forward.
    """
    logging.warning("Baseline results reset.")
    run_models("records/configs", BASELINE_DIR, "baseline_pred.conllu")


def run_comparison(prediction_dir: str) -> None:
    """Comparing the newly predicted files to the baseline.

    Args:
        prediction_dir: the directory where we are keeping the predicted non-baseline files
    """
    for current_file in glob.iglob(f"{prediction_dir}/*"):
        name_prefix = re.match(r".{2,3}_", os.path.basename(current_file)).group(0)
        # getting baseline
        baselines_files = glob.glob(f"{BASELINE_DIR}/{name_prefix}*")
        if not baselines_files or len(baselines_files) > 1:
            raise MisconfigurationError(
                f"Either 0 or more than one baseline file "
                f'matches the test file prefix "{name_prefix}".'
            )
        baseline_file = baselines_files[0]
        with open(current_file) as f1, open(baseline_file) as f2:
            diff = list(difflib.unified_diff(f1.readlines(), f2.readlines(), n=1))
        if not diff:
            continue
        else:
            logging.info("DIFFS:")
            logging.info("\n".join(diff))
            raise TestFailedError(
                "The files predicted by the current model differ from the baseline"
            )


if __name__ == "__main__":
    # reset_baselines() # break glass in case of emergency
    try:
        os.mkdir(PREDICTIONS_DIR)
    except FileExistsError:
        pass
    # running current version
    run_models("records/configs", PREDICTIONS_DIR, "current_pred.conllu")
    run_comparison(PREDICTIONS_DIR)
    shutil.rmtree(PREDICTIONS_DIR)
