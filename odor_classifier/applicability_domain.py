"""k-NN distance-based applicability domain (AD).

Standard QSAR approach (Jaworska/Netzeva-style): for each training
compound, compute the mean distance to its k nearest neighbours in the
(scaled, selected) descriptor space. The cutoff is a chosen percentile of
that training-distance distribution. A new compound is "in domain" if its
mean distance to its k nearest training neighbours does not exceed the
cutoff -- i.e. it sits in a region of chemical space the model actually
learned from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .config import Config, DEFAULT_CONFIG


@dataclass
class ApplicabilityDomain:
    config: Config = field(default_factory=lambda: DEFAULT_CONFIG)
    _nn: Optional[NearestNeighbors] = field(default=None, repr=False)
    cutoff_: Optional[float] = field(default=None, repr=False)
    train_distances_: Optional[np.ndarray] = field(default=None, repr=False)

    def fit(self, X_train: pd.DataFrame) -> "ApplicabilityDomain":
        k = self.config.ad_k_neighbors
        self._nn = NearestNeighbors(n_neighbors=k + 1)  # +1: a training point is its own nearest neighbour
        self._nn.fit(X_train.values)

        distances, _ = self._nn.kneighbors(X_train.values)
        mean_dist = distances[:, 1:].mean(axis=1)  # drop self-distance (col 0)
        self.train_distances_ = mean_dist
        self.cutoff_ = float(np.percentile(mean_dist, self.config.ad_percentile))
        return self

    def mean_distance(self, X: pd.DataFrame) -> np.ndarray:
        if self._nn is None:
            raise RuntimeError("ApplicabilityDomain must be fit before use.")
        k = self.config.ad_k_neighbors
        distances, _ = self._nn.kneighbors(X.values, n_neighbors=k)
        return distances.mean(axis=1)

    def in_domain(self, X: pd.DataFrame) -> np.ndarray:
        return self.mean_distance(X) <= self.cutoff_

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ApplicabilityDomain":
        return joblib.load(path)
