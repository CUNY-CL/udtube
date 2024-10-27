import difflib
import os
import shutil
import unittest

from parameterized import parameterized
from udtube_package.UDTube.cli import udtube_python_interface

CONFIGS_PATH = "records/configs"
BASELINE_DIR = "datasets/baseline_predictions"
PREDICTIONS_DIR = "datasets/current_predictions"
NUM_BASELINES = 3


class UDTubePerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            os.mkdir(PREDICTIONS_DIR)
        except FileExistsError:
            pass

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("records/tmp/")
        shutil.rmtree(PREDICTIONS_DIR)

    @staticmethod
    def _run_model(yaml_config: str, prediction_file_path: str) -> None:
        udtube_python_interface(
            [
                "fit",
                f"--config={yaml_config}",
                f"--prediction.path={prediction_file_path}",
            ]
        )
        udtube_python_interface(
            [
                "predict",
                f"--config={yaml_config}",
                f"--prediction.path={prediction_file_path}",
            ]
        )

    @staticmethod
    def _run_file_comparison(prediction_file: str, baseline_file: str) -> list:
        with open(prediction_file) as f1, open(baseline_file) as f2:
            diff = list(
                difflib.unified_diff(
                    f1.readlines(),
                    f2.readlines(),
                    fromfile=prediction_file,
                    tofile=baseline_file,
                    n=1,
                )
            )
        return diff

    def test_validate_directory(self) -> None:
        """Validate if we can run tests on it here. The goal is to try to catch any silly directory management errors"""
        self.assertTrue(
            os.path.exists(CONFIGS_PATH) and os.listdir(CONFIGS_PATH),
            msg=f"The configs directory {CONFIGS_PATH} must not be empty. Was something deleted?",
        )
        self.assertTrue(
            os.path.exists(BASELINE_DIR) and os.listdir(BASELINE_DIR),
            msg=f"The baseline directory {BASELINE_DIR} must not be empty. Was something deleted?",
        )
        self.assertTrue(
            len(os.listdir(BASELINE_DIR)) == NUM_BASELINES,
            msg=f"The baseline directory {BASELINE_DIR} contains a different number of files than "
            f"expected. {NUM_BASELINES} is the expected amount",
        )

    @parameterized.expand([
        "en",
        "ru",
        "el",
    ])
    def test_model(self, language_code: str):
        config = f"{CONFIGS_PATH}/{language_code}_config.yaml"
        baseline = f"{BASELINE_DIR}/{language_code}_baseline_pred.conllu"
        pred_file_name = f"{PREDICTIONS_DIR}/{language_code}_current_pred.conllu"
        self._run_model(config, pred_file_name)
        diff = self._run_file_comparison(pred_file_name, baseline)
        self.assertEqual(diff, [], msg=f"Differences in the file exist. Diff:\n{diff}")
