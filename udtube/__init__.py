"""UDTube: a neural morphological analyzer.

This module just silences some uninformative warnings.
"""

import os
import warnings

# Silences tokenizers warning about forking.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Silences some stupid warnings.
warnings.filterwarnings("ignore", ".*both args and command line arguments.*")
warnings.filterwarnings("ignore", ".*need to be provided during `Trainer`.*")
warnings.filterwarnings("ignore", ".*is a wandb run already in progress.*")
warnings.filterwarnings("ignore", ".*`tensorboardX` has been removed.*")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*Couldn't infer the batch indices.*")
warnings.filterwarnings("ignore", ".*in eval mode at the start of training.*")
warnings.filterwarnings("ignore", ".*smaller than the logging interval.*")
