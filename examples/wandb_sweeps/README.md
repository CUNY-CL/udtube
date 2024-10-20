# W&B Sweeps

This directory contains example scripts for running a hyperparameter sweep with
[Weights & Biases](https://wandb.ai/site).

-   [`grid.yaml`](grid.yaml) contains just one possible hyperparameter grid,
    designed for random search; edit that file directly to build a
    hyperparameter grid appropriate for your problem.
-   Consider also running Bayesian search (`method: bayes`) instead of random
    search; see
    [here](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#configuration-keys)
    for more information.
-   When running [`sweep.py`](sweep.py) you must provide the `--entity`,
    `--project` and `--sweep_id`; arguments not being tuned should be passed
    using the `--config` file but note that hyperparameters set by the sweep
    will override those specified in the `--config`.
-   By default `random` and `bayes` search run indefinitely, until they are
    killed. To specify a fixed number of samples, provide the `--count` argument
    to [`sweep.py`](sweep.py).

For more information about W&B sweeps, [read
here](https://docs.wandb.ai/guides/sweeps).

## Usage

Execute the following to create and run the sweep; here `${ENTITY}` and
`${PROJECT}` are assumed to be pre-specified environmental variables.

    # Creates a sweep; save the sweep ID as ${SWEEP_ID} for later.
    wandb sweep --entity "${ENTITY}" --project "${PROJECT}" grid.yaml
    # Runs the sweep itself using hyperparameters from the the sweep and
    # additional fixed parameters from a UDTube config file.
    ./sweep.py --entity "${ENTITY}" --project "${PROJECT}" \
         --sweep_id "${SWEEP_ID}" --count "${COUNT}" --config syntagrus.yaml

Then, one can retrieve the results as follows:

1.  Visit the following URL:
    `https://wandb.ai/${ENTITY}/${PROJECT}/sweeps/${SWEEP_ID}`

2.  Switch to "table view" by either clicking on the spreadsheet icon in the top
    left or typing Ctrl+J.

3.  Click on the downward arrow link, select "CSV Export", then click "Save as
    CSV".
