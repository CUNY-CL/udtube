"""Tests system behavior with control characters classified as whitespace."""

import os
import tempfile
import unittest

from udtube import cli

# Directory the unit test is located in, relative to the working directory.
DIR = os.path.relpath(os.path.dirname(__file__), os.getcwd())
CONFIG_PATH = os.path.join(DIR, "testdata/udtube_config.yaml")
TESTDATA_DIR = os.path.join(DIR, "testdata")


class ControlTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="udtube_test-")
        self.assertNonEmptyFileExists(CONFIG_PATH)
        self.encoder = "distilbert/distilbert-base-uncased"

    def assertNonEmptyFileExists(self, path: str):
        self.assertTrue(os.path.exists(path), msg=f"file {path} not found")
        self.assertGreater(
            os.stat(path).st_size, 0, msg="file {path} is empty"
        )

    def test_control_token(self):
        # Testing a case where there's a whitespace token in the sentence.
        # Keeping the test as minimal as possible;
        # train/eval/pred are the same, we are more concerned about the file.
        data_path = os.path.join(TESTDATA_DIR, "badsentence.conllu")
        model_dir = os.path.join(self.tempdir.name, "models")
        cli.udtube_python_interface(
            [
                "fit",
                f"--config={CONFIG_PATH}",
                f"--data.model_dir={model_dir}",
                f"--data.train={data_path}",
                f"--data.val={data_path}",
                f"--model.encoder={self.encoder}",
            ]
        )
        # Confirms a checkpoint was created.
        checkpoint_path = (
            f"{model_dir}/lightning_logs/version_0/checkpoints/last.ckpt"
        )
        self.assertNonEmptyFileExists(checkpoint_path)
        predicted_path = os.path.join(
            self.tempdir.name, "badsentence_predicted.conllu"
        )
        # Predicts on "expected" data.
        cli.udtube_python_interface(
            [
                "predict",
                f"--ckpt_path={checkpoint_path}",
                f"--config={CONFIG_PATH}",
                f"--data.model_dir={model_dir}",
                f"--data.predict={data_path}",
                f"--prediction.path={predicted_path}",
                f"--model.encoder={self.encoder}",
            ]
        )
        # There was a bug where this caused system exit before file writing.
        self.assertNonEmptyFileExists(predicted_path)


if __name__ == "__main__":
    unittest.main()
