"""Held-out audit must never mine test labels or use a production-refit model."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from odor_classifier.heldout_label_audit import (
    save_heldout_audit_snapshot, run_saved_heldout_audit,
    evaluate_heldout_recommendations, INPUT_FOLDER,
)
from odor_classifier.prediction_label_audit import AuditOptions


CLASSES = ['fruity','floral','sweet','woody']


def _make_snapshot(tmp_path):
    train_smiles = ['C' * (2+i) + 'O' for i in range(22)] + [
        'c1ccccc1','Cc1ccccc1','CCc1ccccc1','CCCc1ccccc1','COC(C)=O','CCOC(C)=O',
    ]
    test_smiles = ['CCCCC(=O)O','COCCOC','Oc1ccccc1','CC(C)O']
    y_train = np.array([[1,1,1,0]]*22 + [[0,0,0,1]]*6, dtype=np.uint8)
    y_test = np.array([[1,1,1,0],[1,0,1,0],[0,0,0,1],[1,1,0,0]],dtype=np.uint8)
    probabilities = np.array([
        [.88,.80,.28,.01], [.70,.32,.41,.14], [.07,.11,.11,.87], [.83,.21,.45,.12],
    ])
    thresholds = np.full(4,.5)
    train_x = np.array([[i*.15, float(i%3), float(i%5)] for i in range(28)])
    test_x = np.array([[1.,0,1],[2.,1,2],[3.,2,2],[1.,1,0]])
    save_heldout_audit_snapshot(
        tmp_path, class_names=CLASSES, train_smiles=train_smiles,
        test_smiles=test_smiles, y_train=y_train,y_test=y_test,
        probabilities=probabilities,predictions=probabilities>=thresholds,
        thresholds=thresholds,train_descriptors=train_x,test_descriptors=test_x,
        in_domain=np.array([True]*4),ad_mean_distances=np.array([1.,1.,1.,1.]),
        ad_cutoff=10.0,model_name='tuned_catboost',
        train_names=[f'Train {i}' for i in range(28)],test_names=[f'Test {i}' for i in range(4)],
    )
    return train_smiles, test_smiles


def _options():
    return AuditOptions(k_neighbors=5,min_close_neighbors=2,min_tanimoto=0.05,
                        min_support_count=2,min_support_fraction=0,
                        min_rule_lift=1.0)


def test_heldout_snapshot_and_both_audit_modes_are_consistent(tmp_path):
    train_smiles, test_smiles = _make_snapshot(tmp_path)
    assert len(pd.read_csv(tmp_path/INPUT_FOLDER/'training_only_reference.csv')) == 28
    assert len(pd.read_csv(tmp_path/INPUT_FOLDER/'heldout_test_predictions.csv')) == 4
    assert len(pd.read_csv(tmp_path/INPUT_FOLDER/'heldout_test_truth_FOR_EVALUATION_ONLY.csv')) == 4
    # Poison existing production-named files; held-out audit must not read them.
    (tmp_path/'training_ground_truth_binary.csv').write_text('this is not the reference')
    results = run_saved_heldout_audit(tmp_path,knn_representations='both',options=_options())
    assert set(results) == {'descriptor','morgan'}
    for mode,result in results.items():
        assert len(result.compounds) == 4
        assert len(result.decisions) == 4*len(CLASSES)
        assert not result.compounds.training_structure_overlap.any()
        assert result.settings['model_fitted_on'] == 'train_plus_validation_only'
        assert result.settings['arm_reference_compounds'] == 28
        assert (tmp_path/'heldout_label_audit'/mode/'audit_review_suggestions.csv').exists()
        metrics = pd.read_csv(tmp_path/'heldout_label_audit'/mode/'reference_agreement_summary.csv')
        assert set(metrics.audit_task) == {'suspected_overprediction', 'suggested_addition',
                                           'structural_candidate'}
        assert (metrics.total_reference_mismatches >= 0).all()
    assert (tmp_path/'heldout_label_audit'/'knn_comparison_summary.csv').exists()


def test_test_ground_truth_cannot_change_generated_recommendations(tmp_path):
    _make_snapshot(tmp_path)
    opts=_options()
    run_saved_heldout_audit(tmp_path,knn_representations='morgan',options=opts)
    output = tmp_path/'heldout_label_audit'/'morgan'/'audit_all_label_decisions.csv'
    frozen = output.read_bytes()
    truth_file = tmp_path/INPUT_FOLDER/'heldout_test_truth_FOR_EVALUATION_ONLY.csv'
    truth = pd.read_csv(truth_file)
    for cls in CLASSES:
        truth['label__'+cls] = 1-truth['label__'+cls]
    truth.to_csv(truth_file,index=False)
    run_saved_heldout_audit(tmp_path,knn_representations='morgan',options=opts)
    assert output.read_bytes() == frozen


def test_reference_identity_must_match_snapshot(tmp_path):
    _make_snapshot(tmp_path)
    reference_file = tmp_path/INPUT_FOLDER/'training_only_reference.csv'
    ref = pd.read_csv(reference_file)
    ref.loc[0,'SMILES'] = 'C#N'
    ref.to_csv(reference_file,index=False)
    with pytest.raises(ValueError,match='SMILES order or reference identity'):
        run_saved_heldout_audit(tmp_path,options=_options())


def test_missing_snapshot_requires_new_internal_training_run(tmp_path):
    with pytest.raises(FileNotFoundError,match='Rerun the multilabel TRAINING pipeline'):
        run_saved_heldout_audit(tmp_path)


def test_snapshot_rejects_misaligned_binary_outputs(tmp_path):
    with pytest.raises(ValueError,match='disagree'):
        save_heldout_audit_snapshot(
            tmp_path, class_names=CLASSES,train_smiles=['CCO'],test_smiles=['CCC'],
            y_train=np.array([[1,0,0,0]]),y_test=np.array([[1,0,0,0]]),
            probabilities=np.array([[.2,.2,.2,.2]]),
            predictions=np.array([[1,0,0,0]]),thresholds=.5,
            train_descriptors=np.ones((1,3)),test_descriptors=np.ones((1,3)),
            in_domain=np.array([True]),ad_mean_distances=np.array([.1]),ad_cutoff=1.,
            model_name='test')
