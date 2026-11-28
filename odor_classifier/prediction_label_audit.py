"""Post-prediction, multilabel odor audit for genuinely external compounds.

Sources of evidence:
  (1) saved production model per-class probabilities and decision thresholds;
  (2) selected/scaled model-descriptor Euclidean KNN (default), OR
      stereochemistry-aware Morgan/Tanimoto KNN (comparison baseline);
  (3) positive association rules learned only from reference label transactions.

No external ground-truth labels are needed or read.  This tool produces
human-review suggestions, never modifications of predicted or original labels.
Rule confidence and weighted neighbor prevalence are empirical frequencies,
NOT calibrated probabilities of a correct annotation.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from math import ceil, sqrt
import hashlib
from pathlib import Path
import json
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.neighbors import NearestNeighbors

from .arm_fpgrowth import fp_growth_itemsets


@dataclass(frozen=True)
class AuditOptions:
    # "descriptor" matches the model/AD feature space; "morgan" is the baseline.
    knn_representation: str = 'descriptor'
    k_neighbors: int = 10
    # The AD cutoff is a mean k-NN training distance, NOT a nearest-neighbor
    # cutoff. This factor is a heuristic for qualifying individual neighbors.
    descriptor_distance_factor: float = 1.0
    min_tanimoto: float = 0.40
    min_close_neighbors: int = 3
    low_neighbor_support: float = 0.20
    high_neighbor_support: float = 0.65
    min_support_count: int = 20
    min_support_fraction: float = 0.005
    min_rule_confidence: float = 0.30
    min_rule_lift: float = 1.20
    max_antecedents: int = 3
    add_rule_confidence: float = 0.45
    # Not a calibrated probability: near-threshold model support for a
    # structurally supported unpredicted odor.
    addition_model_fraction: float = 0.65

    def validate(self):
        if self.knn_representation not in ('descriptor', 'morgan'):
            raise ValueError(
                "knn_representation must be 'descriptor' or 'morgan'.")
        if not np.isfinite(self.descriptor_distance_factor) or self.descriptor_distance_factor <= 0:
            raise ValueError(
                'descriptor_distance_factor must be finite and positive.')
        if self.k_neighbors < 1 or self.min_close_neighbors < 1:
            raise ValueError(
                'k_neighbors and min_close_neighbors must be positive.')
        if self.min_close_neighbors > self.k_neighbors:
            raise ValueError('min_close_neighbors cannot exceed k_neighbors.')
        if not (0 <= self.min_tanimoto <= 1):
            raise ValueError('min_tanimoto must be between 0 and 1.')
        if not (0 <= self.low_neighbor_support < self.high_neighbor_support <= 1):
            raise ValueError(
                'Neighbor support cutoffs must satisfy 0 <= low < high <= 1.')
        if not (1 <= self.max_antecedents <= 7):
            raise ValueError('max_antecedents must be between 1 and 7.')
        if self.min_support_count < 1 or not (0 <= self.min_support_fraction <= 1):
            raise ValueError(
                'Support thresholds must be nonnegative (count >=1).')
        if not (0 <= self.min_rule_confidence <= 1) or self.min_rule_lift < 0:
            raise ValueError('Invalid association rule filters.')
        if not (0 <= self.add_rule_confidence <= 1):
            raise ValueError('add_rule_confidence must be between 0 and 1.')
        if not (0 <= self.addition_model_fraction <= 1):
            raise ValueError(
                'addition_model_fraction must be between 0 and 1.')


@dataclass
class AuditResult:
    compounds: pd.DataFrame
    decisions: pd.DataFrame
    neighbors: pd.DataFrame
    rules: pd.DataFrame
    settings: Optional[dict] = None

    @property
    def suggestions(self) -> pd.DataFrame:
        return self.decisions[self.decisions['action'].isin(('review_add', 'review_predicted', 'structural_candidate'))].copy()

    def save(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.compounds.to_csv(out / 'audit_compounds.csv', index=False)
        self.decisions.to_csv(
            out / 'audit_all_label_decisions.csv', index=False)
        self.suggestions.to_csv(
            out / 'audit_review_suggestions.csv', index=False)
        self.neighbors.to_csv(out / 'audit_nearest_neighbors.csv', index=False)
        self.rules.to_csv(out / 'audit_mined_rules.csv', index=False)
        if self.settings is not None:
            (out / 'audit_run_parameters.json').write_text(
                json.dumps(self.settings, indent=2, sort_keys=True), encoding='utf-8')


def _make_fp(mol, generator):
    return None if mol is None else generator.GetFingerprint(mol)


def _canonical_smiles(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles).strip())
    return (None, None) if mol is None else (Chem.MolToSmiles(mol, isomericSmiles=True), mol)


def _wilson_lower(success: int, total: int, z: float = 1.95996398454) -> float:
    if total == 0:
        return 0.0
    p = success / total
    d = 1 + z * z / total
    return (p + z*z/(2*total) - z * sqrt((p*(1-p)+z*z/(4*total))/total)) / d


def mine_label_rules(y: np.ndarray, class_names: list[str], options: AuditOptions) -> pd.DataFrame:
    """Mine exact frequent itemsets via FP-Growth and directional one-label rules.

    Itemsets are unordered. Support counts include every reference compound
    containing the itemset, even with additional labels. The same support,
    confidence, lift and ranking filters are retained from the prior miner.
    External labels and model predictions are never used for mining.
    """
    options.validate()
    y = np.asarray(y, dtype=bool)
    if y.ndim != 2 or y.shape[1] != len(class_names) or y.shape[0] == 0:
        raise ValueError('Reference matrix has an invalid shape.')
    max_size = min(y.shape[1], options.max_antecedents + 1)
    n = y.shape[0]
    required = max(options.min_support_count, ceil(
        options.min_support_fraction * n))
    # Genuine FP-tree mining; all subsets of a frequent itemset are also
    # frequent, so their exact antecedent and singleton counts are present.
    counts = fp_growth_itemsets(y, min_count=required, max_len=max_size)
    records = []
    for itemset, joint in counts.items():
        if len(itemset) < 2 or joint < required:
            continue
        for consequent_index in itemset:
            ante = tuple(x for x in itemset if x != consequent_index)
            antecedent_n = counts[ante]
            consequent_n = counts[(consequent_index,)]
            confidence = joint / antecedent_n
            lift = confidence / (consequent_n / n) if consequent_n else np.nan
            if confidence < options.min_rule_confidence or lift < options.min_rule_lift:
                continue
            prior = consequent_n / n
            shorter = [counts[tuple(sorted(s + (consequent_index,)))] / counts[s]
                       for k in range(1, len(ante)) for s in combinations(ante, k)
                       if counts[s] > 0]
            incremental_gain = confidence - max(shorter, default=prior)
            records.append({
                'antecedent': ', '.join(class_names[j] for j in ante),
                'consequent': class_names[consequent_index],
                'antecedent_size': len(ante),
                'antecedent_count': int(antecedent_n),
                'joint_count': int(joint),
                'support': joint / n,
                'confidence': confidence,
                'confidence_lower95': _wilson_lower(joint, antecedent_n),
                'lift': lift,
                'consequent_prevalence': prior,
                'incremental_gain': incremental_gain,
                '_antecedent_indices': ante,
            })
    cols = ['antecedent', 'consequent', 'antecedent_size', 'antecedent_count', 'joint_count',
            'support', 'confidence', 'confidence_lower95', 'lift', 'consequent_prevalence',
            'incremental_gain', '_antecedent_indices']
    if not records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame.from_records(records, columns=cols).sort_values(
        ['confidence_lower95', 'incremental_gain', 'joint_count', 'lift'],
        ascending=[False, False, False, False], kind='stable', ignore_index=True)


class TrainingReference:
    """Reference set of *production-training* compounds and their saved targets."""

    def __init__(self, reference_df: pd.DataFrame, class_names: list[str],
                 *, radius: int = 2, n_bits: int = 2048, use_chirality: bool = True):
        expected = ['label__' + name for name in class_names]
        if 'SMILES' not in reference_df or any(c not in reference_df for c in expected):
            raise ValueError('Reference file must contain SMILES and every model label__<class> target column. '
                             'Use outputs/reports/training_ground_truth_binary.csv from the final training run.')
        if len(set(class_names)) != len(class_names):
            raise ValueError('Duplicate class names in model artifact.')
        ydf = reference_df[expected].apply(pd.to_numeric, errors='raise')
        if ydf.isna().any().any() or (~ydf.isin([0, 1])).any().any():
            raise ValueError(
                'Reference label columns must contain only 0/1 values.')
        if 'Name' in reference_df:
            names = reference_df['Name'].fillna('').astype(str).tolist()
        else:
            names = [''] * len(reference_df)
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=int(radius), fpSize=int(n_bits), includeChirality=bool(use_chirality))
        self.class_names = list(class_names)
        self.total_reference_rows = len(reference_df)
        self.original_smiles = reference_df['SMILES'].astype(str).tolist()
        self.smiles = []
        self.canonical = []
        self.names = []
        fps = []
        valid_indices = []
        for i, raw in enumerate(reference_df['SMILES']):
            canon, mol = _canonical_smiles(raw)
            if mol is None:
                continue
            valid_indices.append(i)
            self.smiles.append(str(raw))
            self.canonical.append(canon)
            self.names.append(names[i])
            fps.append(_make_fp(mol, self.generator))
        if not fps:
            raise ValueError('No parseable SMILES in training reference.')
        self.valid_indices = np.asarray(valid_indices, dtype=int)
        self.canonical_to_indices = defaultdict(list)
        for j, canon in enumerate(self.canonical):
            self.canonical_to_indices[canon].append(j)
        self.fingerprints = fps
        self.Y = ydf.to_numpy(dtype=np.uint8)[valid_indices]
        self.label_prior = self.Y.mean(axis=0)

    @classmethod
    def from_csv(cls, path: str | Path, class_names: list[str], **kwargs):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError('Training reference not found: ' + str(path) + '. '
                                    'Run the multilabel training pipeline first; do not substitute external test labels.')
        frame = pd.read_csv(path, low_memory=False)
        return cls(frame, class_names, **kwargs)


def training_descriptors_from_ad(ad, reference: TrainingReference, *,
                                 expected_columns=None) -> np.ndarray:
    """Retrieve production model-space *training* features without refitting.

    The production pipeline fits its AD on ``X_full`` in the same order as
    ``training_ground_truth_binary.csv``. sklearn NearestNeighbors retains
    exactly that fitted matrix. This works with existing saved AD artifacts.
    Because ``_fit_X`` is a private sklearn attribute, fail explicitly if
    unavailable instead of silently recomputing a mismatched representation.
    """
    nn = getattr(ad, '_nn', None)
    values = getattr(nn, '_fit_X', None)
    if values is None:
        raise ValueError('Saved AD artifact does not expose its fitted training features. '
                         'Retrain and persist training features or use Morgan KNN.')
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError(
            'AD fitted training matrix is not a finite 2D matrix.')
    if len(x) != reference.total_reference_rows:
        raise ValueError(f'AD/reference row mismatch: {len(x)} fitted AD rows versus '
                         f'{reference.total_reference_rows} reference rows. '
                         'Use AD and training_ground_truth_binary.csv from the same production run.')
    expected_hash = getattr(ad, 'training_smiles_sha256_', None)
    if expected_hash is not None:
        actual_hash = hashlib.sha256(
            '\n'.join(reference.original_smiles).encode('utf-8')).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError('The training reference SMILES or row order does not match '
                             'the saved production AD. Load both from the same training run.')
    if expected_columns is not None and x.shape[1] != len(expected_columns):
        raise ValueError(f'AD/query descriptor width mismatch: {x.shape[1]} versus '
                         f'{len(expected_columns)}. Load the same saved feature pipeline and AD.')
    return x[reference.valid_indices].copy()


def _validated_descriptors(arr, n_rows, *, description, n_features=None) -> np.ndarray:
    values = np.asarray(arr, dtype=float)
    if values.ndim != 2 or values.shape[0] != n_rows:
        raise ValueError(f'{description} must be a matrix with {n_rows} rows.')
    if n_features is not None and values.shape[1] != n_features:
        raise ValueError(
            f'{description} has {values.shape[1]} columns, expected {n_features}.')
    if not np.isfinite(values).all():
        raise ValueError(
            f'{description} contains missing/nonfinite descriptors.')
    return values


def _max_applicable_rule(rules: pd.DataFrame, label: str, present: set[str]):
    if rules.empty:
        return None
    subset = rules[rules['consequent'] == label]
    best = None
    for _, rule in subset.iterrows():
        if set(s.strip() for s in rule['antecedent'].split(',')) <= present:
            best = rule
            break  # records already ranked by lower bound, gain and count
    return best


def audit_predictions(
    smiles_list: list[str],
    probabilities: np.ndarray,
    class_names: list[str],
    reference: TrainingReference,
    *,
    thresholds: Optional[np.ndarray] = None,
    predicted_binary: Optional[np.ndarray] = None,
    in_domain: Optional[np.ndarray] = None,
    ad_mean_distances: Optional[np.ndarray] = None,
    ad_cutoff: Optional[float] = None,
    options: AuditOptions = AuditOptions(),
    rules: Optional[pd.DataFrame] = None,
    training_descriptors: Optional[np.ndarray] = None,
    query_descriptors: Optional[np.ndarray] = None,
    descriptor_distance_cutoff: Optional[float] = None,
) -> AuditResult:
    """Evaluate every compound/label without using external true labels.

    A compound with too few close neighbors in the chosen representation receives
    `insufficient_evidence`, not a forced correction. All flags are proposals
    for human review; the model's original predictions are not modified.
    """
    options.validate()
    n, m = len(smiles_list), len(class_names)
    if n == 0:
        raise ValueError('At least one external compound is required.')
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (n, m) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError(
            'Expected finite per-class model probabilities of shape (N, K) in [0, 1].')
    if list(class_names) != reference.class_names:
        raise ValueError('Class order differs from the training reference.')
    cutoff_values = np.asarray(
        0.5 if thresholds is None else thresholds, dtype=float)
    cutoff_values = np.broadcast_to(cutoff_values, (m,))
    if np.any((cutoff_values < 0) | (cutoff_values > 1)):
        raise ValueError('Model thresholds must be between 0 and 1.')
    pred = p >= cutoff_values if predicted_binary is None else np.asarray(
        predicted_binary, dtype=bool)
    if pred.shape != (n, m):
        raise ValueError('Invalid predicted_binary shape.')
    ad = np.full(n, True) if in_domain is None else np.asarray(
        in_domain, dtype=bool)
    if ad.shape != (n,):
        raise ValueError('in_domain must have one element per compound.')
    distances = np.full(n, np.nan) if ad_mean_distances is None else np.asarray(
        ad_mean_distances, dtype=float)
    if distances.shape != (n,):
        raise ValueError(
            'ad_mean_distances must have one element per compound.')
    rules = mine_label_rules(reference.Y, class_names,
                             options) if rules is None else rules
    # Fit the descriptor neighbor index ONCE for the complete reference. The
    # values are already transformed by the saved production FeaturePipeline.
    if options.knn_representation == 'descriptor':
        if training_descriptors is None or query_descriptors is None:
            raise ValueError('Descriptor KNN needs saved AD training descriptors and the '
                             'prediction-time transformed features. No new scaler fit is allowed.')
        x_train = _validated_descriptors(training_descriptors, len(reference.Y),
                                         description='Training descriptors')
        x_query = _validated_descriptors(query_descriptors, n,
                                         description='Query descriptors', n_features=x_train.shape[1])
        if (descriptor_distance_cutoff is None or
                not np.isfinite(descriptor_distance_cutoff) or descriptor_distance_cutoff <= 0):
            raise ValueError('descriptor_distance_cutoff must be positive. '
                             'Use the fitted production AD cutoff times the chosen factor.')
        nn = NearestNeighbors(metric='euclidean', n_neighbors=min(
            options.k_neighbors + 5, len(x_train)))
        nn.fit(x_train)
        # Extra neighbors ensure that we can remove any exact training
        # structures while still retrieving k genuinely independent neighbors.
        candidate_distances, candidate_indices = nn.kneighbors(x_query)
        descriptor_scale = float(
            descriptor_distance_cutoff) / options.descriptor_distance_factor
    # Pre-index valid rules once: avoid thousands of pandas filters in the label loop.
    rule_index = defaultdict(list)
    for rule in rules.to_dict('records'):
        rule_index[rule['consequent']].append(
            (frozenset(x.strip() for x in rule['antecedent'].split(',')), rule))
    summary_rows, decision_rows, neighbors_rows = [], [], []
    for i, raw_smiles in enumerate(smiles_list):
        canonical, molecule = _canonical_smiles(raw_smiles)
        predicted = {class_names[j] for j in np.flatnonzero(pred[i])}
        max_sim = np.nan
        close_count = 0
        training_overlap = False
        neigh_idxs = np.array([], dtype=int)
        neigh_sims = np.array([], dtype=float)
        nearest_descriptor_distance = np.nan
        neighbor_distance_cutoff = np.nan
        if molecule is not None:
            training_overlap = canonical in reference.canonical_to_indices
            overlap = set(reference.canonical_to_indices.get(canonical, []))
            if options.knn_representation == 'morgan':
                fp = reference.generator.GetFingerprint(molecule)
                similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(
                    fp, reference.fingerprints), dtype=float)
                for idx in overlap:
                    similarities[idx] = -1.0
                order = np.argsort(-similarities, kind='stable')
                order = order[similarities[order] >= 0][:options.k_neighbors]
                metric_values = similarities[order]
                neigh_sims = metric_values
                if len(order):
                    max_sim = float(metric_values[0])
                close_mask = metric_values >= options.min_tanimoto
                neighbor_weights = metric_values[close_mask]
            else:
                # Descriptor distances calculated in the EXACT model feature
                # space (including any PCA). No fitting to external data.
                candidate_rows = [
                    (int(idx), float(dist))
                    for idx, dist in zip(candidate_indices[i], candidate_distances[i])
                    if int(idx) not in overlap
                ][:options.k_neighbors]
                order = np.asarray(
                    [idx for idx, _ in candidate_rows], dtype=int)
                metric_values = np.asarray(
                    [dist for _, dist in candidate_rows], dtype=float)
                if len(metric_values):
                    nearest_descriptor_distance = float(metric_values[0])
                neighbor_distance_cutoff = float(descriptor_distance_cutoff)
                close_mask = metric_values <= neighbor_distance_cutoff
                # Smooth distance-decay weights (not similarities or calibrated
                # probabilities). Scale is the saved AD mean-distance cutoff.
                neighbor_weights = np.exp(
                    -metric_values[close_mask] / max(descriptor_scale, 1e-12))
            close_ids = order[close_mask]
            close_count = len(close_ids)
            close_y = reference.Y[close_ids] if close_count else np.zeros(
                (0, m), dtype=np.uint8)
            if close_count and neighbor_weights.sum() > 0:
                support = np.average(close_y, axis=0, weights=neighbor_weights)
            else:
                support = np.full(m, np.nan)
            close_pos_count = close_y.sum(axis=0).astype(
                int) if close_count else np.zeros(m, dtype=int)
            for rank, (idx, val, accepted) in enumerate(zip(order, metric_values, close_mask), 1):
                neighbors_rows.append({
                    'row_number': i+1, 'SMILES': str(raw_smiles),
                    'rank': rank, 'training_SMILES': reference.smiles[idx],
                    'training_name': reference.names[idx],
                    'knn_representation': options.knn_representation,
                    'euclidean_distance': round(float(val), 6) if options.knn_representation == 'descriptor' else np.nan,
                    'tanimoto': round(float(val), 6) if options.knn_representation == 'morgan' else np.nan,
                    'neighbor_weight': round(float(np.exp(-val / max(descriptor_scale, 1e-12))) if options.knn_representation == 'descriptor'
                                             else float(val), 6),
                    'is_close': bool(accepted),
                    'training_labels': ', '.join(class_names[j] for j in np.flatnonzero(reference.Y[idx]))})
        else:
            support = np.full(m, np.nan)
            close_pos_count = np.zeros(m, dtype=int)

        evidence_reliable = close_count >= options.min_close_neighbors
        additions, questions, possible_additions = [], [], []
        for j, label in enumerate(class_names):
            has_label = bool(pred[i, j])
            other_predicted = predicted - {label}
            rule = next((r for ante, r in rule_index.get(
                label, []) if ante <= other_predicted), None)
            rule_conf = float(rule['confidence']
                              ) if rule is not None else np.nan
            rule_lift = float(rule['lift']) if rule is not None else np.nan
            rule_count = int(rule['joint_count']) if rule is not None else 0
            arm_good = bool(rule is not None and rule_conf >=
                            options.add_rule_confidence)
            structural = float(support[j]) if np.isfinite(
                support[j]) else np.nan
            action = 'no_change'
            rationale = 'No sufficiently strong disagreement between evidence sources.'
            if training_overlap:
                action = 'training_overlap'
                rationale = 'Exact canonical structure in production training set; not an independent external compound.'
            elif not evidence_reliable:
                action = 'insufficient_evidence'
                rationale = ('Too few sufficiently similar training molecules; '
                             'absence of neighbor support is not evidence against this label.')
            elif has_label:
                # A missing positive ARM rule is NOT evidence against a label.
                if structural <= options.low_neighbor_support:
                    action = 'review_predicted'
                    questions.append(label)
                    rationale = ('Predicted positive, but few sufficiently similar training '
                                 'molecules carry this odor; review for possible overprediction.')
                else:
                    action = 'retain'
                    rationale = 'No strong structural contradiction to the predicted label.'
            else:
                if structural >= options.high_neighbor_support and int(close_pos_count[j]) >= options.min_close_neighbors:
                    # Evidence from model near-cutoff OR a reliable ARM rule.
                    model_near = p[i, j] >= options.addition_model_fraction * \
                        cutoff_values[j]
                    if arm_good:
                        action = 'review_add'
                        additions.append(label)
                        rationale = ('Not predicted, but well-supported by similar reference molecules; '
                                     'association rule or near-threshold model score adds context.')
                    else:
                        action = 'structural_candidate'
                        possible_additions.append(label)
                        rationale = ('Strong structural support, but neither model nor a retained '
                                     'association rule supports addition strongly enough.')
            # Heuristic *sorting* indicator; never claim calibrated likelihood.
            contradiction = (1-structural) if has_label else structural
            heuristic_priority = (round(100 * float(contradiction), 1)
                                  if evidence_reliable and np.isfinite(contradiction) else np.nan)
            decision_rows.append({
                'row_number': i+1, 'SMILES': str(raw_smiles), 'label': label,
                'model_predicted': has_label, 'model_probability': float(p[i, j]),
                'model_threshold': float(cutoff_values[j]),
                'reference_label_prevalence': float(reference.label_prior[j]),
                'weighted_knn_support': structural,
                'knn_representation': options.knn_representation,
                'nearest_descriptor_distance': nearest_descriptor_distance,
                'neighbor_distance_cutoff': neighbor_distance_cutoff,
                'close_neighbors': close_count,
                'close_neighbors_with_label': int(close_pos_count[j]),
                'best_rule_antecedent': (str(rule['antecedent']) if rule is not None else ''),
                'rule_confidence': rule_conf, 'rule_lift': rule_lift,
                'rule_joint_count': rule_count,
                'in_domain': bool(ad[i]), 'train_structure_overlap': training_overlap,
                'max_tanimoto': max_sim,
                'action': action, 'review_priority_heuristic': heuristic_priority,
                'reason': rationale,
            })
        status = ('training_overlap' if training_overlap else ('review' if additions or questions else (
            'insufficient_evidence' if not evidence_reliable else ('low_priority_candidate' if possible_additions else 'no_high_priority_flag'))))
        summary_rows.append({
            'row_number': i+1, 'SMILES': str(raw_smiles),
            'predicted_labels': ', '.join(class_names[j] for j in np.flatnonzero(pred[i])),
            'in_domain': bool(ad[i]), 'ad_mean_distance': float(distances[i]),
            'ad_cutoff': float(ad_cutoff) if ad_cutoff is not None else np.nan,
            'max_tanimoto': max_sim, 'close_neighbors': close_count,
            'knn_representation': options.knn_representation,
            'nearest_descriptor_distance': nearest_descriptor_distance,
            'neighbor_distance_cutoff': neighbor_distance_cutoff,
            'training_structure_overlap': training_overlap,
            'review_add': ', '.join(additions),
            'review_predicted': ', '.join(questions),
            'structural_candidates': ', '.join(possible_additions),
            'audit_status': status,
            'note': ('The same structure exists in the training reference; excluded from KNN evidence. '
                     if training_overlap else '') +
            ('Outside model descriptor-space AD. ' if not ad[i] else '') +
            ('Invalid SMILES.' if molecule is None else ''),
        })
    decisions = pd.DataFrame(decision_rows)
    neighbor_columns = ['row_number', 'SMILES', 'rank', 'training_SMILES', 'training_name',
                        'knn_representation', 'euclidean_distance', 'tanimoto',
                        'neighbor_weight', 'is_close', 'training_labels']
    return AuditResult(pd.DataFrame(summary_rows), decisions,
                       pd.DataFrame(neighbors_rows, columns=neighbor_columns),
                       rules.drop(
                           columns=['_antecedent_indices'], errors='ignore'),
                       settings={'arm_algorithm': 'FP-Growth (FP-tree, conditional pattern bases)',
                                 'arm_reference_compounds': int(len(reference.Y)),
                                 'class_names': list(class_names),
                                 'audit_options': asdict(options),
                                 'knn_distance_weighting': ('exp(-distance / ad_cutoff)'
                                                            if options.knn_representation == 'descriptor'
                                                            else 'tanimoto'),
                                 'descriptor_neighbor_cutoff': (float(descriptor_distance_cutoff)
                                                                if options.knn_representation == 'descriptor' else None),
                                 'external_reference_labels_used': False})


def compare_knn_audits(descriptor: AuditResult, morgan: AuditResult):
    """Compare *unmodified* audit decisions from the two KNN representations.

    These tables show agreement, disagreements and coverage, NOT accuracy:
    only independent reference annotations could establish correctness.
    The underlying FP-Growth rules and model probabilities must be identical.
    """
    first = descriptor.decisions
    second = morgan.decisions
    keys = ['row_number', 'SMILES', 'label']
    cols = keys + ['action', 'weighted_knn_support', 'close_neighbors',
                   'model_predicted', 'model_probability']
    frame = first[cols].merge(second[cols], on=keys, how='outer',
                              suffixes=('_descriptor', '_morgan'), validate='one_to_one',
                              indicator=True)
    if not (frame['_merge'] == 'both').all():
        raise ValueError(
            'The two audit reports use different compounds/classes.')
    if not np.allclose(frame.model_probability_descriptor,
                       frame.model_probability_morgan):
        raise ValueError(
            'The KNN runs do not share identical model predictions.')
    frame['same_action'] = frame.action_descriptor == frame.action_morgan
    summary_rows = []
    for name, result in [('descriptor', descriptor), ('morgan', morgan)]:
        decisions = result.decisions
        action_counts = decisions['action'].value_counts()
        summary_rows.append({
            'knn_representation': name,
            'compounds': int(len(result.compounds)),
            'compounds_with_review': int((result.compounds.audit_status == 'review').sum()),
            'compounds_insufficient_evidence': int((result.compounds.audit_status == 'insufficient_evidence').sum()),
            'review_add': int(action_counts.get('review_add', 0)),
            'review_predicted': int(action_counts.get('review_predicted', 0)),
            'structural_candidate': int(action_counts.get('structural_candidate', 0)),
            'all_label_decisions': int(len(decisions)),
            'identical_decisions_between_modes': int(frame.same_action.sum()),
            'different_decisions_between_modes': int((~frame.same_action).sum()),
        })
    return pd.DataFrame(summary_rows), frame.drop(columns='_merge')
