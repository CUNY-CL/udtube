"""Tests trying to create a UDTube model without any classification heads."""

import unittest

from udtube import models, modules


class NoHeadTest(unittest.TestCase):

    def test_no_head_raises_error(self):
        with self.assertRaises(modules.Error):
            # Uses all default parameters except the heads.
            _ = models.UDTube(
                use_upos=False,
                use_xpos=False,
                use_lemma=False,
                use_feats=False,
                use_parse=False,
            )


if __name__ == "__main__":
    unittest.main()
