import glob
import logging
import os
import re
import shutil

from udtube_package.UDTube.cli import udtube_python_interface

BASELINE_DIR = "datasets/baseline_predictions"
BASELINE_SUFFIX = "baseline_pred.conllu"
NUM_BASELINES = 3


class MisconfigurationError(Exception):
    pass


def pretest_validations(configs_path):
    """
    Given a config path, we validate if we can run tests on it here. The goal is to try to catch
    any silly directory management errors
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
        udtube_python_interface(["fit", f"--config={yaml_config}"])
        udtube_python_interface(
            [
                "predict",
                f"--config={yaml_config}",
                f"--prediction.path={prediction_path}",
            ]
        )
    except Exception as e:
        # doing this mainly to propagate the error (finally block must occur no matter what)
        raise e
    finally:
        # clean tmp path, but leave empty folder for convenience
        # TODO - can probably just shutil.rmtree() tmp
        for f in glob.iglob("records/tmp/*"):
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)


def run_models(configs_path: str, prediction_dir: str) -> None:
    """
    Train and predict using several configs
    Args:
        configs_path: path to where the config.yaml files that correspond to baselines are
        prediction_dir: path to where the predictions will be written to
    Returns:
        N/A writes to prediction dir
    """
    for yaml_config in glob.iglob(f"{configs_path}/*"):
        # minor str processing to generate prediction file names
        name_prefix = re.match(r".+_", os.path.basename(yaml_config)).group(0)
        pred_file_name = f"{prediction_dir}/{name_prefix}{BASELINE_SUFFIX}"
        logging.info(
            f"running {yaml_config} and recording test results to {pred_file_name}"
        )
        _run_model(yaml_config, pred_file_name)


def reset_baselines() -> None:
    """
    Reset the baseline to compare tests to.
     Should only be done in rare circumstances when we expect a change in results going forward.
    """
    logging.warning("Baseline results reset.")
    run_models("records/configs", BASELINE_DIR)


if __name__ == "__main__":
    # reset_baselines()
    pass
