"""Mol2vec molecular embeddings using the pretrained Word2Vec substructure model.

Note: ``mol2vec.features.sentences2vec`` from the upstream ``mol2vec``
package targets gensim 3.x (``model.wv.vocab`` / ``word_vec``), which no
longer exist in gensim 4.x. We reimplement the same "sum of substructure
vectors" logic against the modern gensim API instead of depending on it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from gensim.models import word2vec
from rdkit import Chem

from .config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, word2vec.Word2Vec] = {}


def load_mol2vec_model(path: Path | str) -> word2vec.Word2Vec:
    path = str(path)
    if path not in _MODEL_CACHE:
        logger.info("Loading mol2vec model from %s", path)
        _MODEL_CACHE[path] = word2vec.Word2Vec.load(path)
    return _MODEL_CACHE[path]


def _sentence_to_vector(sentence: List[str], model: word2vec.Word2Vec, unseen_vec: Optional[np.ndarray]) -> np.ndarray:
    vocab = model.wv.key_to_index
    vectors = []
    for identifier in sentence:
        if identifier in vocab:
            vectors.append(model.wv.get_vector(identifier))
        elif unseen_vec is not None:
            vectors.append(unseen_vec)
    if not vectors:
        return np.zeros(model.wv.vector_size, dtype=np.float32)
    return np.sum(vectors, axis=0)


def compute_mol2vec_embeddings(
    mols: List[Optional[Chem.Mol]], config: Config = DEFAULT_CONFIG
) -> pd.DataFrame:
    """One embedding vector per molecule (sum of pretrained substructure vectors)."""
    from mol2vec.features import mol2alt_sentence

    model = load_mol2vec_model(config.mol2vec_model_path)
    dim = model.wv.vector_size
    unseen_vec = model.wv.get_vector("UNK") if "UNK" in model.wv.key_to_index else None

    rows = np.zeros((len(mols), dim), dtype=np.float32)
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        sentence = mol2alt_sentence(mol, config.mol2vec_radius)
        rows[i] = _sentence_to_vector(sentence, model, unseen_vec)

    cols = [f"mol2vec_{i}" for i in range(dim)]
    return pd.DataFrame(rows, columns=cols)
