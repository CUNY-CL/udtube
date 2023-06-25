import logging

from torch.utils.data import IterableDataset, Dataset
from sklearn.preprocessing import LabelEncoder
import conllu
import logging

LOGGER = logging.getLogger('UDtube')
LOGGER.setLevel(logging.DEBUG)
PAD_TAG = "[PAD]"
UPOS_CLASSES = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
    "_",
    PAD_TAG
]


class ConlluMapDataset(Dataset):
    """Conllu format Dataset. It loads the entire dataset into memory and therefore is only suitable for smaller
    datasets """

    def __init__(self, conllu_file):
        self.conllu_file = conllu_file
        # setting up label encoders
        self.upos_class_encoder = LabelEncoder()
        self.feats_class_encoder = LabelEncoder()
        self.lemma_class_encoder = LabelEncoder()
        self.feats_classes = self._get_all_classes('feats')
        self.lemma_classes = self._get_all_classes('lemma')
        self._fit_label_encoders()
        self.data_set = self._get_data()

    def _fit_label_encoders(self):
        self.upos_class_encoder.fit(UPOS_CLASSES)
        # TODO how should I encode feats?
        # self.feats_class_encoder.fit(self.feats_classes)
        self.lemma_class_encoder.fit(self.lemma_classes)

    def _get_all_classes(self, label_name: str):
        """helper function to get all the classes observed in the training set"""
        possible_labels = []
        with open(self.conllu_file) as src:
            d = conllu.parse_incr(src)
            for token_list in d:
                for tok in token_list:
                    if tok[label_name] not in possible_labels:
                        possible_labels.append(tok[label_name])
        # reshape is required to use one hot encoder later
        return possible_labels

    def _get_data(self):
        """Loads in the conllu data and prepares it as a list dataset so that it can be indexed for __getitem__"""
        data_set = []
        LOGGER.info("Building dataset")
        with open(self.conllu_file) as src:
            sent_generator = conllu.parse_incr(src)
            for token_list in sent_generator:
                tokens = []
                uposes = []
                lemmas = []
                for token in token_list:
                    # feats = token["feats"]
                    tokens.append(token["form"])
                    uposes.append(token["upos"])
                    lemmas.append(token["lemma"])
                try:
                    uposes = list(self.upos_class_encoder.transform(uposes))
                except ValueError:
                    for i, upos in enumerate(uposes):
                        if upos.strip() in UPOS_CLASSES:
                            uposes.append(upos)
                        else:
                            LOGGER.warning(f"UPOS: {upos} for form {tokens[i]} not found by label encoder, tagging as "
                                         f"'X' (other) upos")
                            uposes.append("X")
                lemmas = list(self.lemma_class_encoder.transform(lemmas))
                data_set.append(
                    (
                        tokens,
                        uposes,
                        lemmas
                    )
                )
        return data_set

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]


class ConlluIterDataset(IterableDataset):
    pass