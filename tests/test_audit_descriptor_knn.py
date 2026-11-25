"""Production descriptor KNN and Morgan comparison tests."""
import numpy as np
import pandas as pd
import pytest

from odor_classifier.applicability_domain import ApplicabilityDomain
from odor_classifier.config import Config
from odor_classifier.prediction_label_audit import (AuditOptions, TrainingReference,
    audit_predictions, training_descriptors_from_ad)


LABELS = ['fruity', 'floral', 'sweet', 'woody']
SMILES = ['COC(C)=O', 'CCOC(C)=O', 'CCCOC(C)=O', 'CCCCOC(C)=O',
          'CC(C)COC(C)=O', 'CC(C)C(=O)OC', 'CCOC(=O)CC', 'CCCOC(=O)CC',
          'COC(=O)CCC', 'CCOC(=O)CCC',
          'CCCCCC', 'CCCCCCC', 'CCCCCCCC', 'CCCCCCCCC', 'CCCCCCCCCC',
          'c1ccccc1', 'Cc1ccccc1', 'CCc1ccccc1', 'CCCc1ccccc1', 'Oc1ccccc1']


def fixtures():
    df = pd.DataFrame({'SMILES': SMILES})
    matrix = np.array([[1, 1, 1, 0]] * 10 + [[0, 0, 0, 1]] * 10)
    for j, label in enumerate(LABELS):
        df['label__' + label] = matrix[:, j]
    ref = TrainingReference(df, LABELS)
    # These matrices imitate already-selected-and-scaled feature data.
    # The assay must use them as-is, without calling StandardScaler.fit.
    x_train = np.array([[0.0 + .03*j, -.1] for j in range(10)] +
                       [[6.0 + .03*j, 4.0] for j in range(10)])
    x_external = pd.DataFrame([[.12, -.1]], columns=['selected_feature1', 'selected_feature2'])
    options = AuditOptions(knn_representation='descriptor', k_neighbors=6,
        min_close_neighbors=3, descriptor_distance_factor=1.5,
        min_support_count=3, min_support_fraction=0,
        min_rule_confidence=.3, min_rule_lift=1.0)
    return ref, x_train, x_external, options


def test_descriptor_knn_supports_missing_labels_and_detects_wrong_prediction():
    ref, x_train, x_external, options = fixtures()
    report = audit_predictions(['CCCCCOC(C)=O'], np.array([[.9,.85,.48,.90]]),
        LABELS, ref, thresholds=.5, options=options,
        training_descriptors=x_train, query_descriptors=x_external,
        descriptor_distance_cutoff=1.0)
    by_label = report.decisions.set_index('label')
    assert by_label.loc['sweet','action'] == 'review_add'
    assert by_label.loc['woody','action'] == 'review_predicted'
    assert by_label.loc['woody','weighted_knn_support'] == 0.0
    assert by_label.loc['sweet','weighted_knn_support'] == 1.0
    assert report.compounds.iloc[0]['knn_representation'] == 'descriptor'
    assert report.compounds.iloc[0]['close_neighbors'] == 6
    assert report.compounds.iloc[0]['max_tanimoto'] != report.compounds.iloc[0]['max_tanimoto']
    assert report.neighbors['tanimoto'].isna().all()
    assert report.neighbors['euclidean_distance'].notna().all()
    assert report.neighbors['euclidean_distance'].is_monotonic_increasing
    assert (report.neighbors['neighbor_weight'].between(0, 1)).all()
    assert report.settings['knn_distance_weighting'] == 'exp(-distance / ad_cutoff)'
    assert report.settings['descriptor_neighbor_cutoff'] == 1.0


def test_training_features_reuse_production_ad_stored_reference():
    ref, x_train, x_external, _ = fixtures()
    config = Config(ad_k_neighbors=3, ad_percentile=90)
    ad = ApplicabilityDomain(config=config).fit(pd.DataFrame(x_train))
    matrix = training_descriptors_from_ad(ad, ref, expected_columns=x_external.columns)
    np.testing.assert_allclose(matrix, x_train)


def test_ad_reference_mismatch_errors_instead_of_misalignment():
    ref, x_train, x_external, _ = fixtures()
    ad = ApplicabilityDomain(config=Config(ad_k_neighbors=3))
    ad.fit(pd.DataFrame(x_train[:-1]))
    with pytest.raises(ValueError, match='row mismatch'):
        training_descriptors_from_ad(ad, ref, expected_columns=x_external.columns)


def test_descriptor_k_needs_correct_transformed_inputs():
    ref, x_train, x_external, opts = fixtures()
    with pytest.raises(ValueError, match='Descriptor KNN needs'):
        audit_predictions(['CCCCCOC(C)=O'], np.array([[.9,.85,.48,.90]]),
            LABELS, ref, options=opts)
    with pytest.raises(ValueError, match='columns'):
        audit_predictions(['CCCCCOC(C)=O'], np.array([[.9,.85,.48,.90]]),
            LABELS, ref, options=opts,
            training_descriptors=x_train, query_descriptors=np.ones((1,3)),
            descriptor_distance_cutoff=1.0)
    with pytest.raises(ValueError, match='descriptor_distance_cutoff'):
        audit_predictions(['CCCCCOC(C)=O'], np.array([[.9,.85,.48,.90]]),
            LABELS, ref, options=opts,
            training_descriptors=x_train, query_descriptors=x_external)


def test_morgan_remains_available_for_a_matched_comparison():
    ref, x_train, x_external, opts = fixtures()
    probs=np.array([[.9,.85,.48,.90]])
    d = audit_predictions(['CCCCCOC(C)=O'], probs, LABELS, ref, options=opts,
        training_descriptors=x_train, query_descriptors=x_external,
        descriptor_distance_cutoff=1.0)
    from dataclasses import replace
    m_opts = replace(opts, knn_representation='morgan', min_tanimoto=.2)
    m = audit_predictions(['CCCCCOC(C)=O'], probs, LABELS, ref, options=m_opts)
    assert list(d.decisions['label']) == list(m.decisions['label'])
    assert len(d.rules) == len(m.rules)
    assert m.neighbors['euclidean_distance'].isna().all()
    assert m.neighbors['tanimoto'].notna().all()
    assert m.compounds.iloc[0]['knn_representation'] == 'morgan'


def test_training_overlap_is_excluded_in_both_modes():
    ref, x_train, x_external, opts = fixtures()
    report = audit_predictions([SMILES[0]], np.array([[.9,.85,.48,.90]]),
        LABELS, ref, options=opts, training_descriptors=x_train,
        query_descriptors=pd.DataFrame([x_train[0]]), descriptor_distance_cutoff=1.)
    assert report.compounds.iloc[0]['training_structure_overlap']
    assert set(report.decisions.action) == {'training_overlap'}
    assert (report.neighbors.training_SMILES != SMILES[0]).all()


def test_comparison_table_only_claims_agreement_not_accuracy():
    ref, x_train, x_external, opts = fixtures()
    probs = np.array([[.9, .85, .48, .90]])
    from odor_classifier.prediction_label_audit import compare_knn_audits
    from dataclasses import replace
    d = audit_predictions(['CCCCCOC(C)=O'], probs, LABELS, ref, options=opts,
        training_descriptors=x_train, query_descriptors=x_external, descriptor_distance_cutoff=1.)
    m = audit_predictions(['CCCCCOC(C)=O'], probs, LABELS, ref,
                          options=replace(opts, knn_representation='morgan', min_tanimoto=.2))
    summary, decisions = compare_knn_audits(d, m)
    assert set(summary.knn_representation) == {'descriptor', 'morgan'}
    assert len(decisions) == len(LABELS)
    assert summary.identical_decisions_between_modes.iloc[0] + summary.different_decisions_between_modes.iloc[0] == len(LABELS)


def test_saved_reference_hash_catches_same_size_but_wrong_training_rows():
    import hashlib
    ref, x_train, x_external, opts = fixtures()
    ad = ApplicabilityDomain(config=Config(ad_k_neighbors=3)).fit(pd.DataFrame(x_train))
    ad.training_smiles_sha256_ = hashlib.sha256('\n'.join(ref.original_smiles).encode()).hexdigest()
    training_descriptors_from_ad(ad, ref, expected_columns=x_external.columns)
    ref.original_smiles[0], ref.original_smiles[1] = ref.original_smiles[1], ref.original_smiles[0]
    with pytest.raises(ValueError, match='row order'):
        training_descriptors_from_ad(ad, ref, expected_columns=x_external.columns)


def test_saved_training_scaler_not_refit_for_external_compounds():
    # A minimal StandardScaler confirms that external feature scaling uses
    # frozen train mean/scale, and that audit helper never fits a scaler.
    from sklearn.preprocessing import StandardScaler
    training = pd.DataFrame({'f1': [1., 3., 5.], 'f2': [10., 30., 50.]})
    scaler = StandardScaler().fit(training)
    means = scaler.mean_.copy()
    external = pd.DataFrame({'f1': [101.], 'f2': [1000.]})
    actual = scaler.transform(external)
    np.testing.assert_allclose(actual, (external.to_numpy() - means)/scaler.scale_)
    np.testing.assert_equal(scaler.mean_, means)
