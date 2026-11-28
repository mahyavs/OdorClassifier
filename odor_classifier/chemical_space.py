"""Chemical-space visualization: PCA / t-SNE / UMAP projections of the
compound feature matrix, colored by odor class and labeled with each
compound's object number.

For each method this produces:
  - a coordinates CSV (``<prefix>_<method>_coords.csv``) with every computed
    component (PCA: up to ``config.chem_space_pca_components``; t-SNE/UMAP:
    ``config.chem_space_tsne_umap_components``) plus No./Name/class columns.
    Saved coordinates allow component selection without recomputation.
  - for PCA only, a variance CSV (``<prefix>_PCA_variance.csv``) with each
    component's explained-variance ratio and cumulative variance.
  - a static PNG (matplotlib, first two components) with each point's
    number printed next to it when the dataset is small enough to stay
    readable.
  - a default interactive HTML (Plotly), 3D if at least 3 components were
    computed, where the number/name/class always shows up on hover -- the
    practical way to identify points in a ~6000-point cloud.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

DIM_PREFIX = {"PCA": "PC", "tSNE": "Dim", "UMAP": "Dim"}


def compute_pca_embedding(X: pd.DataFrame, config: Config = DEFAULT_CONFIG) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (coords, explained_variance_ratio)."""
    n_pca = max(2, min(config.chem_space_pca_components, X.shape[1], X.shape[0] - 1))
    pca = PCA(n_components=n_pca, random_state=config.random_state)
    coords = pca.fit_transform(X.values)
    variance_ratio = pca.explained_variance_ratio_
    logger.info(
        "PCA: computed %d components (PC1+PC2 explain %.1f%% of variance).",
        n_pca, variance_ratio[:2].sum() * 100,
    )
    return coords, variance_ratio


def _prereduce_for_manifold(X: pd.DataFrame, config: Config, pca_components: Optional[int] = None) -> np.ndarray:
    """Quick linear PCA pre-reduction before a t-SNE/UMAP fit.

    Both methods fit on pairwise distances, which get noisy and slow
    computed directly in the full ~400-dimensional selected feature space --
    standard practice (and the reasoning behind sklearn's own ``TSNE``
    defaulting to ``init="pca"``, which only seeds the initial 2D/3D
    coordinates and is unrelated to this) is to reduce to a few dozen PCA
    components first. No-op (returns the raw values, unchanged) once the
    matrix is already narrower than the target -- e.g. when
    ``config.use_pca`` already reduced it upstream -- or when the effective
    component count is 0 (disabled). ``pca_components`` overrides
    ``config.tsne_umap_pca_components`` when given,.
    """
    n_components = config.tsne_umap_pca_components if pca_components is None else pca_components
    if n_components <= 0 or X.shape[1] <= n_components:
        return X.values
    n_components = max(1, min(n_components, X.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=config.random_state)
    reduced = pca.fit_transform(X.values)
    logger.info(
        "Pre-reduced %d features to %d PCA components before t-SNE/UMAP fit (%.1f%% variance retained).",
        X.shape[1], n_components, pca.explained_variance_ratio_.sum() * 100,
    )
    return reduced


def compute_tsne_embedding(
    X: pd.DataFrame, config: Config = DEFAULT_CONFIG, perplexity: Optional[float] = None,
    pca_components: Optional[int] = None,
) -> np.ndarray:
    """t-SNE projection. ``perplexity`` overrides ``config.tsne_perplexity`` when given --
    allows parameter exploration without creating a new Config object. Perplexity must stay
    under roughly a third of the sample count regardless of what's requested (t-SNE's own
    constraint, not a stylistic choice), so the effective value is clamped and logged if
    the request had to be reduced. ``pca_components`` overrides
    ``config.tsne_umap_pca_components`` (see :func:`_prereduce_for_manifold`); pass 0 to
    disable PCA pre-reduction and fit directly on the full feature matrix.
    """
    n_manifold = max(2, min(config.chem_space_tsne_umap_components, 3))
    requested = config.tsne_perplexity if perplexity is None else perplexity
    effective = min(requested, max(5.0, (len(X) - 1) / 3))
    if effective != requested:
        logger.info("t-SNE: requested perplexity %.1f exceeds what %d samples support; using %.1f instead.",
                     requested, len(X), effective)
    tsne = TSNE(n_components=n_manifold, random_state=config.random_state, perplexity=effective, init="pca")
    return tsne.fit_transform(_prereduce_for_manifold(X, config, pca_components))


def compute_umap_embedding(
    X: pd.DataFrame, config: Config = DEFAULT_CONFIG,
    min_dist: Optional[float] = None, n_neighbors: Optional[int] = None,
    pca_components: Optional[int] = None,
) -> np.ndarray:
    """UMAP projection. ``min_dist``/``n_neighbors`` override the matching config fields
    when given, for the same on-demand-recompute reason as :func:`compute_tsne_embedding`.
    ``pca_components`` overrides ``config.tsne_umap_pca_components`` the same way.
    """
    import umap

    n_manifold = max(2, min(config.chem_space_tsne_umap_components, 3))
    requested_neighbors = config.umap_n_neighbors if n_neighbors is None else n_neighbors
    reducer = umap.UMAP(
        n_components=n_manifold,
        n_neighbors=min(requested_neighbors, len(X) - 1),
        min_dist=config.umap_min_dist if min_dist is None else min_dist,
        random_state=config.random_state,
    )
    return reducer.fit_transform(_prereduce_for_manifold(X, config, pca_components))


def compute_embeddings(
    X: pd.DataFrame, config: Config = DEFAULT_CONFIG
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    """Multi-component PCA, t-SNE and UMAP projections of the (already scaled) feature matrix.

    Returns (embeddings, pca_variance_ratio) where embeddings maps method
    name -> (n_samples, n_components) array.
    """
    embeddings: Dict[str, np.ndarray] = {}
    embeddings["PCA"], pca_variance_ratio = compute_pca_embedding(X, config)
    embeddings["tSNE"] = compute_tsne_embedding(X, config)
    embeddings["UMAP"] = compute_umap_embedding(X, config)
    return embeddings, pca_variance_ratio


def _axis_label(method_name: str, dim_index: int, variance_ratio: Optional[np.ndarray]) -> str:
    prefix = DIM_PREFIX[method_name]
    if method_name == "PCA" and variance_ratio is not None and dim_index < len(variance_ratio):
        return f"{prefix}{dim_index + 1} ({variance_ratio[dim_index] * 100:.1f}% var.)"
    return f"{prefix}{dim_index + 1}"


def _save_coords_csv(
    coords: np.ndarray,
    labels: pd.Series,
    point_ids: pd.Series,
    names: Optional[pd.Series],
    smiles: Optional[pd.Series],
    method_name: str,
    path: Path,
) -> None:
    prefix = DIM_PREFIX[method_name]
    df = pd.DataFrame({f"{prefix}{i + 1}": coords[:, i] for i in range(coords.shape[1])})
    df.insert(0, "SMILES", smiles.values if smiles is not None else "")
    df.insert(0, "class", labels.values)
    df.insert(0, "Name", names.values if names is not None else "")
    df.insert(0, "No.", point_ids.values)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %s coordinates (%d components) to %s", method_name, coords.shape[1], path)


def _save_pca_variance_csv(variance_ratio: np.ndarray, path: Path) -> None:
    df = pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(len(variance_ratio))],
        "variance_ratio": variance_ratio,
        "cumulative_variance": np.cumsum(variance_ratio),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved PCA explained-variance table to %s", path)


def _plot_static(
    coords: np.ndarray,
    labels: pd.Series,
    point_ids: pd.Series,
    method_name: str,
    path: Path,
    config: Config,
    variance_ratio: Optional[np.ndarray] = None,
) -> None:
    classes = sorted(labels.unique())
    palette = matplotlib.colormaps["tab20"].colors
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(classes)}

    fig, ax = plt.subplots(figsize=(12, 10))
    for c in classes:
        mask = (labels == c).values
        ax.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.75, color=color_map[c], label=str(c))

    if len(coords) <= config.chem_space_max_annotated_points:
        for (x, y), obj_id in zip(coords[:, :2], point_ids):
            ax.annotate(str(obj_id), (x, y), fontsize=5, alpha=0.7, xytext=(2, 2), textcoords="offset points")
    else:
        logger.info(
            "%s: %d points exceeds chem_space_max_annotated_points=%d; "
            "skipping per-point number labels on the static plot (see the interactive HTML instead).",
            method_name, len(coords), config.chem_space_max_annotated_points,
        )

    ax.set_title(f"Chemical space -- {method_name}")
    ax.set_xlabel(_axis_label(method_name, 0, variance_ratio))
    ax.set_ylabel(_axis_label(method_name, 1, variance_ratio))
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, ncol=1, title="class")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    logger.info("Saved static %s plot to %s", method_name, path)


def _plot_interactive(
    coords: np.ndarray,
    labels: pd.Series,
    point_ids: pd.Series,
    names: Optional[pd.Series],
    method_name: str,
    path: Path,
    variance_ratio: Optional[np.ndarray] = None,
) -> None:
    """Default interactive export -- 3D if >=3 components were computed."""
    import plotly.express as px

    df = pd.DataFrame({
        "class": labels.values,
        "No.": point_ids.values,
        "Name": names.values if names is not None else "",
    })
    is_3d = coords.shape[1] >= 3
    axes = ["x", "y", "z"] if is_3d else ["x", "y"]
    for i, axis in enumerate(axes):
        df[axis] = coords[:, i]

    hover = {"No.": True, "Name": True, "class": True, **{a: False for a in axes}}
    labels_map = {axes[i]: _axis_label(method_name, i, variance_ratio) for i in range(len(axes))}
    title = f"Chemical space -- {method_name} ({'3D' if is_3d else '2D'}, hover for compound No. / name)"

    if is_3d:
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="class", hover_data=hover, title=title, labels=labels_map)
        fig.update_traces(marker=dict(size=3, opacity=0.8))
    else:
        fig = px.scatter(df, x="x", y="y", color="class", hover_data=hover, title=title, labels=labels_map)
        fig.update_traces(marker=dict(size=6, opacity=0.8))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))
    logger.info("Saved interactive %s plot to %s", method_name, path)


def generate_chemical_space_plots(
    X: pd.DataFrame,
    labels: pd.Series,
    point_ids: pd.Series,
    names: Optional[pd.Series] = None,
    smiles: Optional[pd.Series] = None,
    config: Config = DEFAULT_CONFIG,
    prefix: str = "chemical_space",
) -> Dict[str, Path]:
    """Compute PCA/t-SNE/UMAP embeddings and persist coordinates + variance +
    default static/interactive plots for each.

    Returns a dict of output name -> path.
    """
    embeddings, pca_variance_ratio = compute_embeddings(X, config)
    labels = pd.Series(labels).reset_index(drop=True)
    point_ids = pd.Series(point_ids).reset_index(drop=True)
    if names is not None:
        names = pd.Series(names).reset_index(drop=True)
    if smiles is not None:
        smiles = pd.Series(smiles).reset_index(drop=True)

    paths: Dict[str, Path] = {}

    variance_path = config.plots_dir / f"{prefix}_PCA_variance.csv"
    _save_pca_variance_csv(pca_variance_ratio, variance_path)
    paths["PCA_variance"] = variance_path

    for method_name, coords in embeddings.items():
        variance_ratio = pca_variance_ratio if method_name == "PCA" else None

        coords_path = config.plots_dir / f"{prefix}_{method_name}_coords.csv"
        _save_coords_csv(coords, labels, point_ids, names, smiles, method_name, coords_path)
        paths[f"{method_name}_coords"] = coords_path

        static_path = config.plots_dir / f"{prefix}_{method_name}.png"
        _plot_static(coords, labels, point_ids, method_name, static_path, config, variance_ratio)
        paths[f"{method_name}_static"] = static_path

        interactive_path = config.plots_dir / f"{prefix}_{method_name}.html"
        _plot_interactive(coords, labels, point_ids, names, method_name, interactive_path, variance_ratio)
        paths[f"{method_name}_interactive"] = interactive_path

    return paths
