# Scripts

This directory contains additional Python scripts related to the UDTube project.
Some of these scripts have dependencies beyond what UDTube requires, which are
listed in [`requirements.txt`](requirements.txt). Before using these scripts,
install UDTube and then run (in this directory):

    pip install -r requirements.txt

-   [`convert_to_um.py`](convert_to_um.py) converts the FEATS column of a
    CoNLL-U format file from [Universal
    Dependencies](https://universaldependencies.org/u/feat/index.html) format to
    [UniMorph](https://unimorph.github.io/schema/) format. The user may wish to
    do this conversion to training and validation files before training. Note
    that this may not work with all languages in the Universal Dependencies
    corpus.
-   TODO: describe the remaining scripts.
