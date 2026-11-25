import numpy as np
import pandas as pd
import pytest

from odor_classifier.config import Config
from odor_classifier.descriptors import compute_rdkit_descriptors, smiles_to_mols
from odor_classifier.features import FeaturePipeline, compute_raw_features
from odor_classifier.fingerprints import compute_maccs_keys, compute_morgan_fingerprints
from odor_classifier.mol2vec_features import compute_mol2vec_embeddings

SAMPLE_SMILES = ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "CCCCCCCC", "invalid_smiles_string"]


def _config():
    c = Config()
    c.use_mordred = False  # keep unit tests fast
    c.use_mol2vec = False
    return c


def test_rdkit_descriptors_shape():
    mols = smiles_to_mols(SAMPLE_SMILES)
    df = compute_rdkit_descriptors(mols)
    assert len(df) == len(SAMPLE_SMILES)
    assert df.iloc[-1].isna().all()  # invalid SMILES -> all NaN row
    assert not df.iloc[0].isna().all()


def test_morgan_and_maccs_shapes():
    config = _config()
    mols = smiles_to_mols(SAMPLE_SMILES)
    morgan = compute_morgan_fingerprints(mols, config)
    maccs = compute_maccs_keys(mols)
    assert morgan.shape == (len(SAMPLE_SMILES), config.morgan_n_bits)
    assert maccs.shape == (len(SAMPLE_SMILES), 167)
    assert set(np.unique(morgan.values)).issubset({0, 1})


def test_mol2vec_embedding_shape():
    if not Config().mol2vec_model_path.is_file():
        pytest.skip("Pretrained mol2vec model not supplied")
    config = _config()
    mols = smiles_to_mols(SAMPLE_SMILES)
    emb = compute_mol2vec_embeddings(mols, config)
    assert emb.shape == (len(SAMPLE_SMILES), 300)
    assert not np.allclose(emb.iloc[0].values, emb.iloc[1].values)


def test_compute_raw_features_concatenates_blocks():
    config = _config()
    raw = compute_raw_features(SAMPLE_SMILES, config)
    assert len(raw) == len(SAMPLE_SMILES)
    assert any(c.startswith("rdkit_") for c in raw.columns)
    assert any(c.startswith("morgan_") for c in raw.columns)
    assert any(c.startswith("maccs_") for c in raw.columns)
    assert not any(c.startswith("mol2vec_") for c in raw.columns)


def test_feature_pipeline_fit_transform_roundtrip():
    config = _config()
    config.max_features_after_selection = 20
    smiles = ["CCO", "CCC", "CCCC", "c1ccccc1", "c1ccccc1C", "CCN", "CCCN", "CCOC", "CCCCl", "CCBr"] * 3
    y = pd.Series((["A", "B"] * 15))
    raw = compute_raw_features(smiles, config)
    pipeline = FeaturePipeline(config=config)
    X = pipeline.fit_transform(raw, y)
    assert X.shape[0] == len(smiles)
    assert X.shape[1] <= config.max_features_after_selection

    X2 = pipeline.transform(raw)
    assert list(X2.columns) == list(X.columns)
    assert np.allclose(X.values, X2.values, atol=1e-6)
