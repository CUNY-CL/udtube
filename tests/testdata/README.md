This directory contains small "toy" samples drawn from Universal Dependencies
data for English and Russian for `udtube_test.py`. The `_train.conllu` files are
used to train and validate the model. The `_expected.conllu` files are the
result of applying the model to the training data, and the `_expected.test`
files give accuracy results. Each file contains ten sentences.

Use [`regenerate`](regenerate) to generate the expected data files.
