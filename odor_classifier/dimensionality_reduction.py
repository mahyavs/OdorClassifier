"""On-demand dimensionality reduction, including PCA, t-SNE, UMAP, Kernel PCA,
Isomap, LLE, MDS, Factor Analysis, ICA and Spectral Embedding.

Each method returns a sample-by-component coordinate array.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Isomap/LLE/MDS in particular are O(n^2) or worse in memory/time; warn
# above this compound count so a slow run doesn't look like a hang.
_SLOW_METHOD_WARN_THRESHOLD = 3000


def _n_components(config: Config, n_components: Optional[int]) -> int:
    return max(2, min(n_components or config.chem_space_tsne_umap_components, 3))


def compute_pca(X: pd.DataFrame, config: Config = DEFAULT_CONFIG, n_components: Optional[int] = None) -> np.ndarray:
    """PCA, capped to 2-3 components for this tab's fixed 2D/3D scatter
    view (chemical_space.compute_pca_embedding instead computes up to
    config.chem_space_pca_components and lets the Chemical space tab pick
    any pair/triple -- a different visualization contract from this tab's
    uniform "n components across every method" control).
    """
    from sklearn.decomposition import PCA

    n = _n_components(config, n_components)
    pca = PCA(n_components=n, random_state=config.random_state)
    return pca.fit_transform(X.values)


def compute_tsne(X: pd.DataFrame, config: Config = DEFAULT_CONFIG, n_components: Optional[int] = None) -> np.ndarray:
    """t-SNE, reusing chemical_space.compute_tsne_embedding with this
    tab's n_components override applied via a temporary config copy (that
    function reads config.chem_space_tsne_umap_components directly rather
    than taking n_components as its own parameter).
    """
    from .chemical_space import compute_tsne_embedding

    n = _n_components(config, n_components)
    return compute_tsne_embedding(X, replace(config, chem_space_tsne_umap_components=n))


def compute_umap(X: pd.DataFrame, config: Config = DEFAULT_CONFIG, n_components: Optional[int] = None) -> np.ndarray:
    """UMAP, same n_components-override approach as compute_tsne above."""
    from .chemical_space import compute_umap_embedding

    n = _n_components(config, n_components)
    return compute_umap_embedding(X, replace(config, chem_space_tsne_umap_components=n))


def compute_kernel_pca(X: pd.DataFrame, config: Config = DEFAULT_CONFIG, n_components: Optional[int] = None) -> np.ndarray:
    """Kernel PCA (RBF kernel) -- a non-linear generalization of PCA that
    can separate clusters a purely linear projection can't.
    """
    from sklearn.decomposition import KernelPCA

    n = _n_components(config, n_components)
    reducer = KernelPCA(n_components=n, kernel="rbf", random_state=config.random_state)
    return reducer.fit_transform(X.values)


def compute_isomap(X: pd.DataFrame, config: Config = DEFAULT_CONFIG, n_components: Optional[int] = None,
                    n_neighbors: Optional[int] = None) -> np.ndarray:
    """Isomap: preserves geodesic (manifold) distances via a neighbor
    graph, rather than Euclidean distance directly -- good at unrolling
    curved manifolds PCA would flatten incorrectly. O(n^2)-ish; slow on
    large compound counts.
    """
    from sklearn.manifold import Isomap

    if len(X) > _SLOW_METHOD_WARN_THRESHOLD:
        logger.warning("Isomap scales poorly with compound count; %d compounds may take a while.", len(X))
    n = _n_components(config, n_components)
    k = min(n_neighbors or config.umap_n_neighbors, len(X) - 1)
    reducer = Isomap(n_components=n, n_neighbors=k)
    return reducer.fit_transform(X.values)


def compute_lle(X: pd.DataFrame, config: Config = DEFAULT_CONFIG, n_components: Optional[int] = None,
                 n_neighbors: Optional[int] = None) -> np.ndarray:
    """Locally Linear Embedding: reconstructs each point as a linear
    combination of its nearest neighbors, then finds a low-dimensional
    embedding preserving those local reconstruction weights. O(n^2)-ish;
    slow on large compound counts.
    """
    from sklearn.manifold import LocallyLinearEmbedding

    if len(X) > _SLOW_METHOD_WARN_THRESHOLD:
        logger.warning("LLE scales poorly with compound count; %d compounds may take a while.", len(X))
    n = _n_components(config, n_components)
    k = min(n_neighbors or config.umap_n_neighbors, len(X) - 1)
    reducer = LocallyLinearEmbedding(n_components=n, n_neighbors=k, random_state=config.random_state)
    return reducer.fit_transform(X.values)


def compute_mds(X: pd.DataFrame, config: Config = DEFAULT_CONFIG, n_components: Optional[int] = None) -> np.ndarray:
    """Multi-Dimensional Scaling: finds a low-dimensional layout whose
    pairwise Euclidean distances best match the original high-dimensional
    pairwise distances. O(n^2) memory (a full distance matrix); slow and
    memory-hungry on large compound counts.
    """
    from sklearn.manifold import MDS

    if len(X) > _SLOW_METHOD_WARN_THRESHOLD:
        logger.warning("MDS scales poorly with compound count (O(n^2) memory); %d compounds may take a while "
                        "or use a lot of memory.", len(X))
    n = _n_components(config, n_components)
    reducer = MDS(n_components=n, random_state=config.random_state, normalized_stress="auto", n_init=1, init="random")
    return reducer.fit_transform(X.values)


def compute_factor_analysis(X: pd.DataFrame, config: Config = DEFAULT_CONFIG, n_components: Optional[int] = None) -> np.ndarray:
    """Factor Analysis: a linear generative model assuming each feature is
    a noisy linear combination of a few shared latent factors -- unlike
    PCA, it explicitly separates shared signal from per-feature noise
    variance rather than treating all variance as signal.
    """
    from sklearn.decomposition import FactorAnalysis

    n = _n_components(config, n_components)
    reducer = FactorAnalysis(n_components=n, random_state=config.random_state)
    return reducer.fit_transform(X.values)


def compute_ica(X: pd.DataFrame, config: Config = DEFAULT_CONFIG, n_components: Optional[int] = None) -> np.ndarray:
    """Independent Component Analysis (FastICA): finds statistically
    independent (not just uncorrelated, as in PCA) latent components --
    useful when the underlying descriptor variation is a mixture of
    distinct, non-Gaussian sources rather than one broad Gaussian spread.
    """
    from sklearn.decomposition import FastICA

    n = _n_components(config, n_components)
    reducer = FastICA(n_components=n, random_state=config.random_state, max_iter=1000)
    return reducer.fit_transform(X.values)


def compute_spectral_embedding(X: pd.DataFrame, config: Config = DEFAULT_CONFIG, n_components: Optional[int] = None,
                                n_neighbors: Optional[int] = None) -> np.ndarray:
    """Spectral Embedding (Laplacian Eigenmaps): builds a nearest-neighbor
    graph and embeds points using the low-frequency eigenvectors of its
    graph Laplacian -- like Isomap/LLE, a manifold-aware alternative to
    linear PCA, but derived from graph spectral theory instead of geodesic
    distance or local linear reconstruction.
    """
    from sklearn.manifold import SpectralEmbedding

    n = _n_components(config, n_components)
    k = min(n_neighbors or config.umap_n_neighbors, len(X) - 1)
    reducer = SpectralEmbedding(n_components=n, n_neighbors=k, random_state=config.random_state)
    return reducer.fit_transform(X.values)


def prepare_unsupervised_features(raw_features: pd.DataFrame, config: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Clean + scale a raw descriptor matrix with no target-based selection
    step -- there's no y to select against in a purely unsupervised
    context. Drops all-NaN columns, median-imputes, variance-filters,
    drops one of each highly-correlated pair, then standard-scales. Used
    when the Dimensionality Reduction tab is pointed at a freshly-uploaded
    Excel file rather than reusing the classifier's already-fitted,
    supervised-selected feature pipeline.
    """
    from sklearn.preprocessing import StandardScaler

    from .features import safe_variance_filter

    df = raw_features.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="all")
    df = df.fillna(df.median(numeric_only=True))
    df = df.dropna(axis=1, how="any")

    df, _ = safe_variance_filter(df, config.variance_threshold)

    if df.shape[1] > 1:
        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if (upper[c] > config.correlation_threshold).any()]
        df = df.drop(columns=to_drop)

    scaled = StandardScaler().fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns, index=df.index)


# Registry of dimensionality-reduction methods.
METHOD_REGISTRY = [
    ("use_pca_method", "PCA", compute_pca),
    ("use_tsne_method", "t-SNE", compute_tsne),
    ("use_umap_method", "UMAP", compute_umap),
    ("use_kernel_pca", "Kernel PCA", compute_kernel_pca),
    ("use_isomap", "Isomap", compute_isomap),
    ("use_lle", "LLE (Locally Linear Embedding)", compute_lle),
    ("use_mds", "MDS (Multi-Dimensional Scaling)", compute_mds),
    ("use_factor_analysis", "Factor Analysis", compute_factor_analysis),
    ("use_ica", "ICA (Independent Component Analysis)", compute_ica),
    ("use_spectral_embedding", "Spectral Embedding", compute_spectral_embedding),
]
