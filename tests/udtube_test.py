"""Full tests of training and prediction.

This runs five epochs of training over a small toy data set, attempting to
overfit, then compares the resubstitution predictions on this set to
previously computed results. As such this is essentially a change-detector
test. Currently, English (en) and Russian (ru) are tested."""

import contextlib
import difflib
import os
import tempfile
import unittest

from parameterized import parameterized

from udtube import cli

# Directory the unit test is located in, relative to the working directory.
DIR = os.path.relpath(os.path.dirname(__file__), os.getcwd())
CONFIG_PATH = os.path.join(DIR, "testdata/udtube_config.yaml")
TESTDATA_DIR = os.path.join(DIR, "testdata")


class UDTubeTest(unittest.TestCase):
    def assertNonEmptyFileExists(self, path: str):
        self.assertTrue(os.path.exists(path), msg=f"file {path} not found")
        self.assertGreater(
            os.stat(path).st_size, 0, msg="file {path} is empty"
        )

    def assertFileIdentity(self, actual_path: str, expected_path: str):
        with (
            open(actual_path, "r") as actual,
            open(expected_path, "r") as expected,
        ):
            diff = list(
                difflib.unified_diff(
                    actual.readlines(),
                    expected.readlines(),
                    fromfile=actual_path,
                    tofile=expected_path,
                    n=1,
                )
            )
        self.assertEqual(diff, [], f"Prediction differences found:\n{diff}")

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="udtube_test-")
        self.assertNonEmptyFileExists(CONFIG_PATH)

    def tearDown(self):
        self.tempdir.cleanup()

    @parameterized.expand(
        [
            ("en", "google-bert/bert-base-cased", True),
            ("ru", "DeepPavlov/rubert-base-cased", False),
        ]
    )
    def test_model(self, langcode: str, encoder: str, use_xpos: bool):
        # Fits model.
        train_path = os.path.join(TESTDATA_DIR, f"{langcode}_train.conllu")
        model_dir = os.path.join(self.tempdir.name, "models")
        cli.udtube_python_interface(
            [
                "fit",
                f"--config={CONFIG_PATH}",
                f"--data.model_dir={model_dir}",
                f"--data.train={train_path}",
                # We are trying to overfit on the training data.
                f"--data.val={train_path}",
                f"--model.encoder={encoder}",
                f"--model.use_xpos={use_xpos}",
            ]
        )
        # Confirms a checkpoint was created.
        checkpoint_path = (
            f"{model_dir}/lightning_logs/version_0/checkpoints/last.ckpt"
        )
        self.assertNonEmptyFileExists(checkpoint_path)
        # Predicts on "expected" data.
        predicted_path = os.path.join(
            self.tempdir.name, f"{langcode}_predicted.conllu"
        )
        expected_path = os.path.join(
            TESTDATA_DIR, f"{langcode}_expected.conllu"
        )
        cli.udtube_python_interface(
            [
                "predict",
                f"--ckpt_path={checkpoint_path}",
                f"--config={CONFIG_PATH}",
                f"--data.model_dir={model_dir}",
                f"--data.predict={expected_path}",
                f"--model.encoder={encoder}",
                f"--model.use_xpos={use_xpos}",
                f"--prediction.path={predicted_path}",
            ]
        )
        self.assertNonEmptyFileExists(predicted_path)
        self.assertFileIdentity(predicted_path, expected_path)
        # Evaluates on "expected" data.
        evaluated_path = os.path.join(
            self.tempdir.name,
            f"{langcode}_evaluated.test",
        )
        with open(evaluated_path, "w") as sink:
            with contextlib.redirect_stdout(sink):
                cli.udtube_python_interface(
                    [
                        "test",
                        f"--ckpt_path={checkpoint_path}",
                        f"--config={CONFIG_PATH}",
                        f"--data.model_dir={model_dir}",
                        f"--data.test={expected_path}",
                        f"--model.encoder={encoder}",
                        f"--model.use_xpos={use_xpos}",
                    ]
                )
        expected_path = os.path.join(
            self.tempdir.name, f"{langcode}_evaluated.test"
        )
        self.assertNonEmptyFileExists(evaluated_path)
        self.assertFileIdentity(evaluated_path, expected_path)


if __name__ == "__main__":
    unittest.main()
