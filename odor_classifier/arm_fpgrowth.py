"""FP-Growth frequent-itemset miner for binary multilabel transactions.

Implements the FP-tree / conditional-pattern-base algorithm of:
Han, Pei & Yin (2000), Mining Frequent Patterns without Candidate Generation.
ACM SIGMOD, doi:10.1145/342009.335372.

The result maps sorted zero-based item-index tuples to *exact* support counts.
It does not discretize, sample, or estimate support. Only itemsets meeting
``min_count`` and ``max_len`` are returned. Transactions may contain any number
of binary positive labels; their order is irrelevant.

This standalone implementation avoids a new runtime dependency on mlxtend.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


class _FPNode:
    __slots__ = ('item', 'count', 'parent', 'children')

    def __init__(self, item: int | None, parent: '_FPNode | None' = None):
        self.item = item
        self.count = 0
        self.parent = parent
        self.children: dict[int, _FPNode] = {}


def _make_tree(weighted_transactions: list[tuple[tuple[int, ...], int]], min_count: int):
    totals = Counter()
    for transaction, weight in weighted_transactions:
        for item in transaction:
            totals[item] += weight
    frequent = {item: n for item, n in totals.items() if n >= min_count}
    if not frequent:
        return {}, {}

    root = _FPNode(None)
    header: dict[int, list[_FPNode]] = defaultdict(list)
    for transaction, weight in weighted_transactions:
        ordered = sorted((item for item in transaction if item in frequent),
                         key=lambda item: (-frequent[item], item))
        current = root
        for item in ordered:
            child = current.children.get(item)
            if child is None:
                child = _FPNode(item, current)
                current.children[item] = child
                header[item].append(child)
            child.count += weight
            current = child
    return frequent, header


def fp_growth_itemsets(y: np.ndarray, min_count: int, max_len: int) -> dict[tuple[int, ...], int]:
    """Return exact support counts for frequent label subsets via FP-Growth.

    Parameters
    ----------
    y : (n_compounds, n_labels) binary matrix
        Each row is an unordered transaction of positive odor labels.
    min_count : int
        Minimum number of reference compounds containing an itemset.
    max_len : int
        Maximum number of labels in any returned itemset.
    """
    y = np.asarray(y)
    if y.ndim != 2 or y.shape[0] == 0:
        raise ValueError('Expected a nonempty 2D binary transaction matrix.')
    if not np.isin(y, (0, 1)).all():
        raise ValueError('FP-Growth requires binary 0/1 label data.')
    if min_count < 1 or max_len < 1:
        raise ValueError('min_count and max_len must be positive integers.')

    transactions = Counter(tuple(int(i) for i in np.flatnonzero(row)) for row in y)
    weighted = [(transaction, multiplicity)
                for transaction, multiplicity in transactions.items() if transaction]
    frequencies, header = _make_tree(weighted, min_count)
    frequent_itemsets: dict[tuple[int, ...], int] = {}

    def grow(freqs: dict[int, int], hdr: dict[int, list[_FPNode]], prefix: tuple[int, ...]):
        # Least-frequent items first: the usual recursive FP-Growth traversal.
        for item in sorted(freqs, key=lambda item: (freqs[item], item)):
            new_prefix = prefix + (item,)
            key = tuple(sorted(new_prefix))
            count = int(freqs[item])
            prior = frequent_itemsets.get(key)
            if prior is not None and prior != count:
                raise AssertionError(f'Inconsistent FP-tree support for itemset {key}.')
            frequent_itemsets[key] = count
            if len(new_prefix) >= max_len:
                continue
            conditional_patterns: list[tuple[tuple[int, ...], int]] = []
            for node in hdr[item]:
                ancestor_items = []
                parent = node.parent
                while parent is not None and parent.item is not None:
                    ancestor_items.append(parent.item)
                    parent = parent.parent
                if ancestor_items:
                    conditional_patterns.append((tuple(ancestor_items), int(node.count)))
            if conditional_patterns:
                next_frequencies, next_header = _make_tree(conditional_patterns, min_count)
                if next_frequencies:
                    grow(next_frequencies, next_header, new_prefix)

    grow(frequencies, header, ())
    return frequent_itemsets
