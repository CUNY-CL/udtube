"""Unit tests for edit_scripts."""

import unittest

from udtube.data import edit_scripts


class EditScriptTest(unittest.TestCase):
    def assertValid(
        self, istring: str, ostring: str, script: edit_scripts.EditScript
    ) -> None:
        # Asserts the edit script produces the expected result.
        exp_ostring = script.apply(istring)
        self.assertEqual(ostring, exp_ostring)

    def test_empty(self):
        token = ""
        script = edit_scripts.EditScript(token, token)
        self.assertEqual(str(script), "")
        self.assertValid(token, token, script)
        rscript = edit_scripts.ReverseEditScript(token, token)
        self.assertEqual(str(rscript), "")
        self.assertValid(token, token, rscript)

    def test_identity(self):
        token = "identify"
        script = edit_scripts.EditScript(token, token)
        self.assertEqual(str(script), "")
        self.assertValid(token, token, script)
        rscript = edit_scripts.ReverseEditScript(token, token)
        self.assertEqual(str(rscript), "")
        self.assertValid(token, token, rscript)
        # Works despite length mismatch.
        token = "idempotent"
        self.assertValid(token, token, script)
        self.assertValid(token, token, rscript)

    def test_prefixation(self):
        before = "umibig"
        after = "ibig"
        script = edit_scripts.EditScript(before, after)
        self.assertValid(before, after, script)

    def test_infixation(self):
        before = "lipad"
        after = "lumipad"
        script = edit_scripts.EditScript(before, after)
        self.assertValid(before, after, script)

    def test_stem_change(self):
        before = "Väter"
        after = "Vater"
        script = edit_scripts.EditScript(before, after)
        self.assertValid(before, after, script)
        rscript = edit_scripts.ReverseEditScript(before, after)
        self.assertValid(before, after, rscript)

    def test_stem_change_with_suffixation(self):
        before = "Bäume"
        after = "Baum"
        rscript = edit_scripts.ReverseEditScript(before, after)
        self.assertValid(before, after, rscript)

    def test_suffixation(self):
        before = "столов"
        after = "стол"
        rscript = edit_scripts.ReverseEditScript(before, after)
        self.assertValid(before, after, rscript)
        # Works despite length mismatch.
        self.assertValid("градусов", "градус", rscript)

    def test_spanish_example(self):
        # After Chrupała et al. 2008, p. 2362.
        before = "pidieron"
        after = "pedir"
        rscript = edit_scripts.ReverseEditScript(before, after)
        self.assertEqual(str(rscript), "~|~||~|||~e")
        self.assertValid(before, after, rscript)
        # Also testing roundtrip serialization/deserialization.
        rscript = edit_scripts.ReverseEditScript.fromtag(str(rscript))
        # Works despite length mismatch.
        self.assertValid("repitieron", "repetir", rscript)


if __name__ == "__main__":
    unittest.main()
