# Scripts

This directory contains additional Python scripts related to the UDTube project.
Some of these scripts have dependencies beyond what UDTube requires, which are
listed in [`requirements.txt`](requirements.txt). Before using these scripts,
install UDTube and then run (in this directory):

    pip install -r requirements.txt

The following scripts are provided:

-   [`convert_to_um.py`](convert_to_um.py) converts the FEATS column of a
    CoNLL-U format file from [Universal
    Dependencies](https://universaldependencies.org/u/feat/index.html) format to
    [UniMorph](https://unimorph.github.io/schema/) format using
    [`ud-compatibility`](https://github.com/unimorph/ud-compatibility). The user
    may wish to do this conversion to training and validation files before
    training. Note that this may not work with all languages in the Universal
    Dependencies corpus.
-   [`evaluate.py`](evaluate.py) provides a general tool for evaluating labeled
    CoNLL-U data; it reports accuracy for language-universal and
    language-specific POS tags, lemmatization, and morphological features.
-   [`remove_mwe.py`](remove_mwe.py) removes multi-word annotations from CoNLL-U
    format files. In practice, the subwords of a multi-word expression are
    usually annotated separately, so this simply cleans up things.
-   [`pretokenize.py`](pretokenize.py) converts raw text into CoNLL-U format
    using [`spacy-udpipe`](https://pypi.org/project/spacy-udpipe/).
-   [`udpipe.py`](updipe.py) applies pretrained UDPipe models using
    [`spacy-udpipe`](https://pypi.org/project/spacy-udpipe/). This is a
    lower-resource alternative to the UDTube pipeline and can be useful for
    comparison.
