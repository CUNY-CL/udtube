"""Tests truncation of long sentences.

This test simply attempts to train on, and then predict the analysis for,
a vey long sentence, one that exceeds the BERT sequence length limitations.

This was reported to fail at #82.
"""

import os
import tempfile
import unittest

from parameterized import parameterized

from udtube import cli

# Directory the unit test is located in, relative to the working directory.
DIR = os.path.relpath(os.path.dirname(__file__), os.getcwd())
CONFIG_PATH = os.path.join(DIR, "testdata/length_config.yaml")
TESTDATA_DIR = os.path.join(DIR, "testdata")


class LengthTest(unittest.TestCase):
    def assertNonEmptyFileExists(self, path: str):
        self.assertTrue(os.path.exists(path), msg=f"file {path} not found")
        self.assertGreater(
            os.stat(path).st_size, 0, msg="file {path} is empty"
        )

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="udtube_test-")
        self.assertNonEmptyFileExists(CONFIG_PATH)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_truncation(self):
        # Fits the model for one epoch.
        model_dir = os.path.join(self.tempdir.name, "models")
        cli.udtube_python_interface(
            [
                "fit",
                f"--config={CONFIG_PATH}",
                f"--data.model_dir={model_dir}",
            ],
        )
        # Confirms a checkpoint was created.
        checkpoint_path = (
            f"{model_dir}/lightning_logs/version_0/checkpoints/last.ckpt"
        )
        self.assertNonEmptyFileExists(checkpoint_path)
        # Attempts prediction.
        cli.udtube_python_interface(
            [
                "predict",
                f"--ckpt_path={checkpoint_path}",
                f"--config={CONFIG_PATH}",
                f"--data.model_dir={model_dir}",
            ]
        )


if __name__ == "__main__":
    unittest.main()
