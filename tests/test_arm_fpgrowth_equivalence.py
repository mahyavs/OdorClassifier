"""FP-Growth exact-count regression tests against independent exhaustive enumeration.

The exhaustive code below is a TEST ORACLE only: production ARM uses an FP-tree.
"""
from collections import Counter
from itertools import combinations
from math import ceil

import numpy as np
import pandas as pd
import pytest

from odor_classifier.arm_fpgrowth import fp_growth_itemsets
from odor_classifier.prediction_label_audit import AuditOptions, mine_label_rules


def _exhaustive_counts(y, min_count, max_size):
    counts = Counter()
    for row in y:
        indices = np.flatnonzero(row).tolist()
        for size in range(1, min(len(indices), max_size) + 1):
            counts.update(combinations(indices, size))
    return {itemset: n for itemset, n in counts.items() if n >= min_count}


@pytest.mark.parametrize('n_rows,n_labels,min_count,max_len', [
    (1, 1, 1, 1), (50, 5, 1, 4), (70, 8, 2, 4),
    (120, 23, 4, 4), (80, 7, 10, 5), (20, 5, 21, 4),
])
def test_fpgrowth_exact_counts_equal_exhaustive(n_rows, n_labels, min_count, max_len):
    rng = np.random.default_rng(20261115 + n_rows + n_labels)
    y = (rng.random((n_rows, n_labels)) < rng.uniform(.05, .7, n_labels)).astype(np.uint8)
    if n_rows > 3:
        y[1] = y[0]  # repeated transaction; FP-tree compression must count twice
        y[2] = 0     # empty transactions must not invent support
    assert fp_growth_itemsets(y, min_count, max_len) == _exhaustive_counts(y, min_count, max_len)


def test_fpgrowth_known_conditional_pattern_counts():
    y = np.array([[1,1,1,0],[1,1,0,0],[1,0,1,0],[0,1,1,0],
                  [1,1,1,0],[0,0,0,0]], dtype=np.uint8)
    frequent = fp_growth_itemsets(y, min_count=2, max_len=3)
    assert frequent[(0,)] == 4
    assert frequent[(0,1)] == 3
    assert frequent[(0,1,2)] == 2
    assert (3,) not in frequent
    assert frequent == _exhaustive_counts(y, 2, 3)


def test_fpgrowth_preserves_original_rule_statistics_and_filter_defaults():
    rng = np.random.default_rng(392)
    y = (rng.random((350, 12)) < rng.uniform(.12, .50, 12)).astype(np.uint8)
    names = [f'label_{i}' for i in range(y.shape[1])]
    options = AuditOptions(min_support_count=8, min_support_fraction=.005,
                           min_rule_confidence=.30, min_rule_lift=1.2)
    new = mine_label_rules(y, names, options)
    # Separate exhaustive reference calculates the same rule statistics using
    # full frequency counts, even for itemsets filtered out by min_count.
    all_counts = _exhaustive_counts(y, 1, options.max_antecedents + 1)
    old_stats = {}
    required = max(options.min_support_count, ceil(options.min_support_fraction * len(y)))
    for items, joint in all_counts.items():
        if len(items) < 2 or joint < required:
            continue
        for consequent in items:
            ante = tuple(x for x in items if x != consequent)
            ante_count = all_counts[ante]
            conf = joint / ante_count
            lift = conf / (all_counts[(consequent,)] / len(y))
            if conf < options.min_rule_confidence or lift < options.min_rule_lift:
                continue
            old_stats[(', '.join(names[i] for i in ante), names[consequent])] = (
                joint, ante_count, joint / len(y), conf, lift)
    new_stats = {(r.antecedent, r.consequent): (
        r.joint_count, r.antecedent_count, r.support, r.confidence, r.lift)
        for r in new.itertuples(index=False)}
    assert new_stats.keys() == old_stats.keys()
    for key in old_stats:
        assert new_stats[key] == pytest.approx(old_stats[key])


def test_fpgrowth_input_validation():
    with pytest.raises(ValueError, match='binary'):
        fp_growth_itemsets(np.array([[1,2],[0,1]]), 1, 2)
    with pytest.raises(ValueError, match='positive'):
        fp_growth_itemsets(np.array([[1,0]]), 0, 2)
