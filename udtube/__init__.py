"""UDTube: a neural morphological analyzer."""

# Silences some uninformative warnings.
import warnings

# Silences irrelevant warnings; these are more like "Did you know?"s.
warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck.*"
)
warnings.filterwarnings(
    "ignore", ".*Couldn't infer the batch indices fetched from.*"
)
