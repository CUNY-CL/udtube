"""Selects the appropriate pretrained encoder from Hugging Face.

This module also includes two types of special-casing:

* In a few casees we need to customize the names of particular arguments;
  currently, dropout is the only argument that requires this.
* We warn the user if they select a pre-trained encoder we haven't tested yet.

Users are encouraged to file pull requests to:

* Add special casing for a pre-trained encoder.
* Add a pre-trained encoder to the "tested" list.
"""

import logging

import transformers


# The keys here are assumed to be prefixes of full name and should include
# the organization name, a forward slash, and the shared prefix of the model.

# These are implicitly included in TESTED_ENCODERS.
# Please keep in lexicographic order.
# The key is the model prefix, the value is the name of the dropout parameter.
SPECIAL_CASE_ENCODERS = {
    "flaubert/flaubert": "dropout",
    "google-t5/t5": "dropout_rate",
}

# These implicitly include the SPECIAL_CASE_ENCODERS.
# Please keep in lexicographic order.
TESTED_ENCODERS = {
    "DeepPavlov/rubert",
    "FacebookAI/xlm-roberta",
    "dccuchile/bert-base-spanish",
    "google-bert/bert",
    "nlpaueb/bert-base-greek",
}


def load(model_name: str, dropout: float) -> transformers.AutoModel:
    """Loads the encoder and applies any special casing.

    Args:
        model_name: the Hugging Face model name.
        dropout: encoder dropout probability.

    Returns:
        A Hugging Face encoder.
    """
    # TODO: If we end up with a very long list of model names we should
    # consider storing them in a prefix tree for faster lookup.
    kwargs = {}
    model_found = False
    # Looks for special-cased encoders.
    for prefix, dropout_name in SPECIAL_CASE_ENCODERS.items():
        if model_name.startswith(prefix):
            kwargs[dropout_name] = dropout
            model_found = False
            break
    # Looks for tested encoders.
    if not model_found:
        for prefix in TESTED_ENCODERS:
            if model_name.startswith(prefix):
                model_found = True
                break
    if not model_found:
        logging.warning(
            "Model %s has not been tested with UDTube; it may require special "
            "casing in %s",
            model_name,
            __file__,
        )
    # Uses this as the default name.
    # TODO: improve this conditional if kwargs not related to dropout are
    # also passed to the encoder loader.
    if not kwargs:
        kwargs["hidden_dropout_prob"] = dropout
    return transformers.AutoModel.from_pretrained(
        model_name, output_hidden_states=True, **kwargs
    )
