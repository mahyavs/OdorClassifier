import numpy as np
import pandas as pd

from odor_classifier.applicability_domain import ApplicabilityDomain
from odor_classifier.config import Config
from odor_classifier.predict import assign_multilabel_with_uc


def test_applicability_domain_flags_outlier():
    config = Config()
    config.ad_k_neighbors = 3
    config.ad_percentile = 90
    rng = np.random.RandomState(0)
    X_train = pd.DataFrame(rng.normal(size=(200, 5)))
    ad = ApplicabilityDomain(config=config).fit(X_train)

    inlier = pd.DataFrame(rng.normal(size=(5, 5)))
    outlier = pd.DataFrame(rng.normal(loc=50, scale=1, size=(5, 5)))

    assert ad.in_domain(inlier).mean() > 0.5
    assert not ad.in_domain(outlier).any()


def test_multilabel_uc_fallback():
    cfg = Config()
    result = assign_multilabel_with_uc(np.array([[.9, .8], [.1, .2], [.9, .8]]),
                                      ["fruity", "sweet"], np.array([True, True, False]), cfg)
    assert result["final_label"].tolist() == ["fruity, sweet", "UC", "UC"]
