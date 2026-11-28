"""odor_classifier: relate chemical compounds (SMILES) to odor classes.

The package predicts multiple odor categories from the 22-class dictionary,
with UC as an unclassified fallback.
"""

from .config import Config, DEFAULT_CONFIG, UC_LABEL

__all__ = ["Config", "DEFAULT_CONFIG", "UC_LABEL"]
__version__ = "0.1.0"
