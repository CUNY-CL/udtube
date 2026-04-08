"""UDTube: a neural morphological analyzer."""

import os
import warnings

# Silences tokenizers warning about forking.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Silences some stupid warnings.
warnings.filterwarnings("ignore", r"(?s).*both args and command line.*")
warnings.filterwarnings("ignore", r"(?s).*need to be provided during.*")
warnings.filterwarnings("ignore", r"(?s).*is a wandb run already in.*")
warnings.filterwarnings("ignore", r"(?s).*`tensorboardX` has been removed.*")
warnings.filterwarnings("ignore", r"(?s).*does not have many workers.*")
warnings.filterwarnings("ignore", r"(?s).*Couldn't infer the batch indices.*")
warnings.filterwarnings("ignore", r"(?s).*in eval mode at the start of.*")
warnings.filterwarnings("ignore", r"(?s).*smaller than the logging interval.*")
