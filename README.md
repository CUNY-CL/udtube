# UDTube (beta)

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/CUNY-CL/udtube/tree/master.svg?style=svg&circle-token=CCIPRJ_4V98VzpnERYSUaGFAkxu7v_70eea48ab82c8f19e4babbaa55a64855a80415bd)](https://dl.circleci.com/status-badge/redirect/gh/CUNY-CL/udtube/tree/master)

UDTube is a neural morphological analyzer based on
[PyTorch](https://pytorch.org/), [Lightning](https://lightning.ai/), and
[Hugging Face transformers](https://huggingface.co/docs/transformers/en/index).

## Philosophy

Named in homage to the venerable
[UDPipe](https://lindat.mff.cuni.cz/services/udpipe/), UDTube is focused on
incremental inference, allowing it to be used to label large text collections.

## Design

The UDTube model consists of a pre-trained (and possibly, fine-tuned)
transformer encoder which feeds into a classifier layer with many as four heads
handling the different morphological tasks.

Lightning is used to generate the [training, validation, inference, and
evaluation
loops](https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#hooks).
The [LightningCLI
interface](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli)
is used to provide a user interface and manage configuration.

Below, we use [YAML](https://yaml.org/) to specify configuration options, and we
strongly recommend users do the same. However, most configuration options can
also be specified using POSIX-style command-line flags.

## Installation

To install UDTube and its dependencies, run the following command:

    pip install .

## File formats

Other than YAML configuration files, most operations use files in
[CoNLL-U](https://universaldependencies.org/format.html) format. This is a
10-column tab-separated format with a blank line between each sentence and `#`
used for comments. In all cases, the `ID` and `FORM` field must be fully
populated; the `_` blank tag can be used for unknown fields.

Many of our experiments are performed using CoNLL-U data from the [Universal
Dependencies project](https://universaldependencies.org/).

## Tasks

UDTube can perform up to four morphological tasks simultaneously:

-   Lemmatization is performed using the `LEMMA` field and [edit
    scripts](https://aclanthology.org/P14-2111/).

-   [Universal part-of-speech
    tagging](https://universaldependencies.org/u/pos/index.html) is performed
    using the `UPOS` field: enable with `data: use_upos: true`.

-   Language-specific part-of-speech tagging is performed using the `XPOS`
    field: enable with `data: use_xpos: true`.

-   Morphological feature tagging is performed using the `FEATS` field: enable
    with `data: use_feats: true`.

The following caveats apply:

-   Note that many newer Universal Dependencies datasets do not have
    language-specific part-of-speech-tags.
-   The `FEATS` field is treated as a single unit and is not segmented in any
    way.
-   One can convert from [Universal Dependencies morphological
    features](https://universaldependencies.org/u/feat/index.html) to [UniMorph
    features](https://unimorph.github.io/schema/) using
    [`scripts/convert_to_um.py`](scripts/convert_to_um.py).
-   UDTube does not perform dependency parsing at present, so the `HEAD`,
    `DEPREL`, and `DEPS` fields are ignored and should be specified as `_`.

## Usage

The `udtube` command-line tool uses a subcommand interface, with the four
following modes. To see the full set of options available with each subcommand,
use the `--print_config` flag. For example:

    udtube fit --print_config

will show all configuration options (and their default values) for the `fit`
subcommand.

### Training (`fit`)

In `fit` mode, one trains a UDTube model from scratch. Naturally, most
configuration options need to be set at training time. E.g., it is not possible
to switch between different pre-trained encoders or enable new tasks after
training.

This mode is invoked using the `fit` subcommand, like so:

    udtube fit --config path/to/config.yaml

#### Seeding

Setting the `seed_everything:` argument to some value ensures a reproducible
experiment.

#### Encoder

The encoder layer consists of a pre-trained BERT-style transformer model. By
default, UDTube uses multilingual cased BERT
(`model: encoder: google-bert/bert-base-multilingual-cased`). In theory, UDTube
can use any Hugging Face pre-trained encoder so long as it provides a
`AutoTokenizer` and has been exposed to the target language. We [list all the
Hugging Face encoders we have tested thus far](udtube/encoders.py), and warn
users when selecting an untested encoder. Since there is no standard for
referring to the between-layer dropout probability parameter, it is in some
cases also necessary to specify what this argument is called for a given model.
We welcome pull requests from users who successfully make use of encoders not
listed here.

So-called "tokenizer-free" pre-trained encoders like ByT5 are not currently
supported as they lack an `AutoTokenizer`.

#### Classifier

The classifier layer contains up to four sequential linear heads for the four
tasks described above. By default all four are enabled.

#### Optimization

UDTube uses separate optimizers and LR schedulers for the encoder and
classifier. The intuition behind this is that we may wish to make slow, small
changes (or possibly, no changes at all) to the pre-trained encoder, whereas we
wish to make more rapid and larger changes to the classifier.

The following YAML snippet shows a simple configuration that encapsulates this
principle. It uses the Adam optimizer for both encoder and classifier, but uses
a lower learning rate for the encoder with a linear warm-up and a higher
learning rate for the classifier.

    ...
    model:
      encoder_optimizer:
        class_path: torch.optim.Adam
        init_args:
          lr: 1e-5
      encoder_scheduler:
        class_path: udtube.schedulers.WarmupInverseSquareRoot
        init_args:
          warmup_epochs: 5
      classifier_optimizer:
        class_path: torch.optim.Adam
        init_args:
          lr: 1e-3
      classifier_scheduler:
        class_path: lightning.pytorch.cli.ReduceLROnPlateau
        init_args:
          monitor: val_loss
          factor: 0.1
      ...

The default scheduler is `udtube.schedulers.DummyScheduler`, which keeps
learning rate fixed to its initial value.

#### Checkpointing

The
[`ModelCheckpoint`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html)
is used to control the generation of checkpoint files. A sample YAML snippet is
given below.

    ...
    checkpoint:
      filename: "model-{epoch:03d}-{val_loss:.4f}"
      monitor: val_loss
      verbose: true
      ...

Without some specification under `checkpoint:` UDTube will not generate
checkpoints!

#### Callbacks

The user will likely want to configure additional callbacks. Some useful
examples are given below.

The
[`LearningRateMonitor`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.LearningRateMonitor.html)
callback records learning rates; this is useful when working with multiple
optimizers and/or schedulers, as we do here. A sample YAML snippet is given
below.

    ...
    trainer:
      callbacks:
      - class_path: lightning.pytorch.callbacks.LearningRateMonitor
        init_args:
          logging_interval: epoch
      ...

The
[`EarlyStopping`](https://lightning.ai/docs/pytorch/stable/common/early_stopping.html)
callback enables early stopping based on a monitored quantity and a fixed
"patience". A sample YAML snipppet with a patience of 10 is given below.

    ...
    trainer:
      callbacks:
      - class_path: lightning.pytorch.callbacks.EarlyStopping
        init_args:
          monitor: val_loss
          patience: 10
          verbose: true
      ...

Adjust the `patience` parameter as needed.

All three of these features are enabled in the [sample configuration
files](configs) we provide.

#### Logging

By default, UDTube performs some minimal logging to standard error and uses
progress bars to keep track of progress during each epoch. However, one can
enable additional logging faculties during training, using a similar syntax to
the one we saw above for callbacks.

The
[`CSVLogger`](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.CSVLogger.html)
logs all monitored quantities to a CSV file. A sample configuration is given
below.

    ...
    trainer:
      logger:
        - class_path: lightning.pytorch.loggers.CSVLogger
          init_args:
            save_dir: /Users/Shinji/models
      ...
       

Adjust the `save_dir` argument as needed.

The
[`WandbLogger`](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html)
works similarly to the `CSVLogger`, but sends the data to the third-party
website [Weights & Biases](https://wandb.ai/site), where it can be used to
generate charts or share artifacts. A sample configuration is given below.

    ...
    trainer:
      logger:
      - class_path: lightning.pytorch.loggers.wandb.WandbLogger
        init_args:
          entity: NERV
          project: unit1
          save_dir: /Users/Shinji/models
      ...

Adjust the `entity`, `project`, and `save_dir` arguments as needed; note that
this functionality requires a working account with Weights & Biases.

#### Other options

By default, UDTube attempts to model all four tasks; one can disable the
language-specific tagging task using `model: use_xpos: false`, and so on.

Dropout probability is specified using `model: dropout: ...`.

The encoder has multiple layers. The input to the classifier consists of just
the last few layers mean-pooled together. The number of layers used for
mean-pooling is specified using `model: pooling_layers: ...`.

By default, lemmatization uses reverse-edit scripts. This is appropriate for
predominantly suffixal languages, which are thought to represent the majority of
the world's languages. If working with a predominantly prefixal language,
disable this with `model: reverse_edits: false`.

The following YAML snippet shows the default architectural arguments.

    ...
    model:
      dropout: 0.5
      encoder: google-bert/bert-base-multilingual-cased
      pooling_layers: 4
      reverse_edits: true
      use_upos: true
      use_xpos: true
      use_lemma: true
      use_feats: true
      ...
      

Batch size is specified using `data: batch_size: ...` and defaults to 32.

There are a number of ways to specify how long a model should train for. For
example, the following YAML snippet specifies that training should run for 100
epochs or 6 wall-clock hours, whichever comes first.

    ...
    trainer:
      max_epochs: 100
      max_time: 00:06:00:00
      ...

### Validation (`validate`)

In `validation` mode, one runs the validation step over labeled validation data
(specified as `data: val: path/to/validation.conllu`) using a previously trained
checkpoint (`--ckpt_path path/to/checkpoint.ckpt` from the command line),
recording total loss and per-task accuracies. In practice this is mostly usefulf
or debugging.

This mode is invoked using the `validate` subcommand, like so:

    udtube validate --config path/to/config.yaml --ckpt_path path/to/checkpoint.ckpt

### Evaluation (`test`)

In `test` mode, we compute accuracy over held-out test data (specified as
`data: test: path/to/test.conllu`) using a previously trained checkpoint
(`--ckpt_path path/to/checkpoint.ckpt` from the command line); it differs from
`validation` mode in that it uses the `test` file rather than the `val` file and
it does not compute loss.

This mode is invoked using the `test` subcommand, like so:

    udtube test --config path/to/config.yaml --ckpt_path path/to/checkpoint.ckpt

### Inference (`predict`)

In `predict` mode, a previously trained model checkpoint
(`--ckpt_path path/to/checkpoint.ckpt` from the command line) is used to label a
CoNLL-U file. One must also specify the path where the predictions will be
written.

    ...
    predict:
      path: /Users/Shinji/predictions.conllu
    ...

Here are some additional details:

-   In `predict` mode UDTube loads the file to be labeled incrementally (i.e.,
    one sentence at a time) so this can be used with very large files.
-   In `predict` mode, if no path for the predictions is specified, stdout will
    be used. If using this in conjunction with \> or \|, add
    `--trainer.enable_progress_bar false` on the command line.
-   The target task fields are overriden if their heads are active.
-   Use [`scripts/pretokenize.py`](scripts/pretokenize.py) to convert raw text
    files to CoNLL-U input files.

This mode is invoked using the `predict` subcommand, like so:

    udtube predict --config path/to/config.yaml --ckpt_path path/to/checkpoint.ckpt

## Examples

See [`examples`](examples/README.md) for some worked examples including
hyperparameter sweeping with [Weights & Biases](https://wandb.ai/site).

## Additional scripts

See [`scripts/README.md`](scripts/README.md) for details on provided scripts not
mention above.

## License

UDTube is distributed under an [Apache 2.0 license](LICENSE.txt).

## Contribution

We welcome contributions using the fork-and-pull model.

## References

If you use UDTube in your research, we would appreciate it if you cited the
following document, which describes the model:

Yakubov, D. 2024. [How do we learn what we cannot
say?](https://academicworks.cuny.edu/gc_etds/5622/) Master's thesis, CUNY
Graduate Center.
