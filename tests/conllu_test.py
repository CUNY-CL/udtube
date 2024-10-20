"""Unit tests for CoNLL-U file parsing."""

import io
import tempfile
import unittest

from udtube import data


class TokenListTest(unittest.TestCase):

    def make_empty_token_list(self) -> data.TokenList:
        return data.TokenList([])

    def make_singleton_token_list(self) -> data.TokenList:
        return data.TokenList(
            [
                {
                    "id": 1,
                    "form": "Hi",
                    "lemma": "hi",
                    "upos": "INTJ",
                    "xpos": "_",
                    "feats": "_",
                    "head": "_",
                    "deprel": "_",
                    "deps": "_",
                    "misc": "_",
                }
            ],
            metadata={"text": "Hi"},
        )

    def test_empty_token_list(self):
        tokens = self.make_empty_token_list()
        self.assertEqual(len(tokens), 0)
        self.assertIsNotNone(tokens.metadata)

    def test_singleton_token_list(self):
        tokens = self.make_singleton_token_list()
        self.assertEqual(len(tokens), 1)
        self.assertIsNotNone(tokens.metadata)

    def test_access(self):
        tokens = self.make_singleton_token_list()
        self.assertEqual(tokens[0]["form"], "Hi")
        tokens.append(
            {
                "id": 2,
                "form": "stranger",
                "lemma": "stranger",
                "upos": "NOUN",
                "xpos": "_",
                "feats": "_",
                "head": "_",
                "deprel": "_",
                "deps": "_",
                "misc": "_",
            },
        )
        tokens.append(
            {
                "id": 3,
                "form": ".",
                "lemma": ".",
                "upos": "PUNCT",
                "xpos": "_",
                "feats": "_",
                "head": "_",
                "deprel": "_",
                "deps": "_",
                "misc": "_",
            },
        )
        tokens.metadata["text"] += " stranger."
        self.assertEqual(len(tokens), 3)
        self.assertEqual(tokens[1]["form"], "stranger")


class ParseTest(unittest.TestCase):

    STRING = """# newpar
# sent_id = weblog-blogspot.com_nominations_20041117172713_ENG_20041117_172713-0001
# text = From the AP comes this story :
1	From	from	ADP	IN	_	3	case	3:case	_
2	the	the	DET	DT	Definite=Def|PronType=Art	3	det	3:det	_
3	AP	AP	PROPN	NNP	Number=Sing	4	obl	4:obl:from	_
4	comes	come	VERB	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	root	0:root	_
5	this	this	DET	DT	Number=Sing|PronType=Dem	6	det	6:det	_
6	story	story	NOUN	NN	Number=Sing	4	nsubj	4:nsubj	_
7	:	:	PUNCT	:	_	4	punct	4:punct	_

# sent_id = weblog-blogspot.com_nominations_20041117172713_ENG_20041117_172713-0002
# text = President Bush on Tuesday nominated two individuals to replace retiring jurists on federal courts in the Washington area.
1	President	President	PROPN	NNP	Number=Sing	5	nsubj	5:nsubj	_
2	Bush	Bush	PROPN	NNP	Number=Sing	1	flat	1:flat	_
3	on	on	ADP	IN	_	4	case	4:case	_
4	Tuesday	Tuesday	PROPN	NNP	Number=Sing	5	obl	5:obl:on	_
5	nominated	nominate	VERB	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	0	root	0:root	_
6	two	two	NUM	CD	NumForm=Word|NumType=Card	7	nummod	7:nummod	_
7	individuals	individual	NOUN	NNS	Number=Plur	5	obj	5:obj	_
8	to	to	PART	TO	_	9	mark	9:mark	_
9	replace	replace	VERB	VB	VerbForm=Inf	5	advcl	5:advcl:to	_
10	retiring	retire	VERB	VBG	VerbForm=Ger	11	amod	11:amod	_
11	jurists	jurist	NOUN	NNS	Number=Plur	9	obj	9:obj	_
12	on	on	ADP	IN	_	14	case	14:case	_
13	federal	federal	ADJ	JJ	Degree=Pos	14	amod	14:amod	_
14	courts	court	NOUN	NNS	Number=Plur	11	nmod	11:nmod:on	_
15	in	in	ADP	IN	_	18	case	18:case	_
16	the	the	DET	DT	Definite=Def|PronType=Art	18	det	18:det	_
17	Washington	Washington	PROPN	NNP	Number=Sing	18	compound	18:compound	_
18	area	area	NOUN	NN	Number=Sing	14	nmod	14:nmod:in	SpaceAfter=No
19	.	.	PUNCT	.	_	5	punct	5:punct	_

"""  # noqa: E501

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = tempfile.NamedTemporaryFile(mode="w", suffix=".conllu")
        print(cls.STRING, file=cls.path)
        cls.path.flush()

    @classmethod
    def tearDownClass(cls):
        cls.path.close()

    def test_parse(self):
        parser = data.parse_from_path(self.path.name)
        s1 = next(parser)
        self.assertIsNone(s1.metadata["newpar"])
        self.assertEqual(s1.metadata["text"], "From the AP comes this story :")
        self.assertEqual(len(s1), 7)
        self.assertEqual(
            [token["form"] for token in s1],
            "From the AP comes this story :".split(),
        )
        s2 = next(parser)
        self.assertEqual(len(s2), 19)
        with self.assertRaises(StopIteration):
            next(parser)

    def test_roundtrip(self):
        buf = io.StringIO()
        for tokenlist in data.parse_from_path(self.path.name):
            print(tokenlist.serialize(), file=buf)
        self.assertEqual(self.STRING, buf.getvalue())


if __name__ == "__main__":
    unittest.main()
