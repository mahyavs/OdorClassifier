import numpy as np
import pandas as pd
import pytest

from odor_classifier.prediction_label_audit import (
    AuditOptions, TrainingReference, audit_predictions, mine_label_rules,
)


CLASSES = ['fruity', 'floral', 'sweet', 'woody']


def _reference():
    smiles = [
        'COC(C)=O', 'CCOC(C)=O', 'CCCOC(C)=O', 'CCCCOC(C)=O',
        'CC(C)COC(C)=O', 'CC(C)C(=O)OC', 'CCOC(=O)CC', 'CCCOC(=O)CC',
        'COC(=O)CCC', 'CCOC(=O)CCC',
        'CCCCCC', 'CCCCCCC', 'CCCCCCCC', 'CCCCCCCCC', 'CCCCCCCCCC',
        'c1ccccc1', 'Cc1ccccc1', 'CCc1ccccc1', 'CCCc1ccccc1', 'Oc1ccccc1',
    ]
    y = np.array([[1,1,1,0]]*10 + [[0,0,0,1]]*10, dtype=np.uint8)
    df = pd.DataFrame({'SMILES':smiles, 'Name':['train'+str(j) for j in range(len(smiles))]})
    for j,name in enumerate(CLASSES):
        df['label__' + name] = y[:,j]
    return TrainingReference(df, CLASSES)


def _opts(**kwargs):
    return AuditOptions(knn_representation='morgan', min_support_count=3, min_support_fraction=0,
                        min_rule_confidence=.3, min_rule_lift=1,
                        min_tanimoto=.20, min_close_neighbors=2,
                        k_neighbors=5, **kwargs)


def test_mining_produces_directional_rules_and_no_label_order():
    ref = _reference()
    rules = mine_label_rules(ref.Y, CLASSES, _opts())
    rule = rules[(rules.antecedent == 'fruity, floral') & (rules.consequent == 'sweet')].iloc[0]
    assert rule.joint_count == 10
    assert rule.antecedent_count == 10
    assert rule.confidence == pytest.approx(1)
    assert rule.lift == pytest.approx(2)


def test_unseen_compounds_audited_without_reference_labels():
    ref = _reference()
    # Unseen ester with fruity/floral predicted and sweet narrowly below threshold.
    test_smiles = ['CCCCCOC(C)=O']
    probs = np.array([[.92, .80, .48, .10]])
    report = audit_predictions(test_smiles, probs, CLASSES, ref, options=_opts(),
                               in_domain=np.array([True]))
    assert len(report.decisions) == 4
    assert len(report.compounds) == 1
    assert report.compounds.iloc[0]['training_structure_overlap'] == False
    assert report.decisions.loc[report.decisions.label=='sweet', 'model_predicted'].iloc[0] == False
    assert 'review_add' in set(report.decisions.action)
    assert 'sweet' in report.compounds.iloc[0]['review_add']


def test_no_sufficient_similarity_must_abstain():
    ref = _reference()
    options = AuditOptions(knn_representation='morgan', min_tanimoto=.99, min_close_neighbors=3,
                           min_support_count=2, min_support_fraction=0)
    report = audit_predictions(['CCCCCOC(C)=O'], np.array([[.92, .80, .48, .10]]),
                               CLASSES, ref, options=options, in_domain=np.array([False]))
    assert set(report.decisions.action) == {'insufficient_evidence'}
    assert report.compounds.iloc[0].audit_status == 'insufficient_evidence'
    assert report.compounds.iloc[0].in_domain == False


def test_training_overlap_excluded_as_neighbor():
    ref = _reference()
    p = np.array([[.8,.8,.8,.2]])
    report = audit_predictions([ref.smiles[0]], p, CLASSES, ref, options=_opts())
    assert report.compounds.iloc[0].training_structure_overlap == True
    assert (report.neighbors.training_SMILES != ref.smiles[0]).all()


def test_bad_ref_or_probability_alignment_rejected():
    ref = _reference()
    with pytest.raises(ValueError, match='shape'):
        audit_predictions(['CCO'], np.array([[.2,.4]]), CLASSES, ref)
    with pytest.raises(ValueError, match='Class order'):
        audit_predictions(['CCO'], np.array([[.2,.4,.5,.6]]), list(reversed(CLASSES)), ref)


def test_structural_contradiction_flags_predicted_label():
    ref = _reference()
    report = audit_predictions(['CCCCCOC(C)=O'], np.array([[.95,.87,.84,.90]]),
                               CLASSES, ref, options=_opts())
    row = report.decisions.set_index('label').loc['woody']
    assert row.action == 'review_predicted'
    assert row.weighted_knn_support == 0
    assert 'woody' in report.compounds.iloc[0].review_predicted


def test_train_overlap_does_not_generate_review_suggestions():
    ref = _reference()
    report = audit_predictions([ref.smiles[0]], np.array([[.8,.8,.1,.9]]),
                               CLASSES, ref, options=_opts())
    assert report.compounds.iloc[0].audit_status == 'training_overlap'
    assert set(report.decisions.action) == {'training_overlap'}
    assert report.suggestions.empty


def test_reading_reference_csv_and_export(tmp_path):
    ref = _reference()
    df = pd.DataFrame({'SMILES':ref.smiles})
    for j,name in enumerate(CLASSES):
        df['label__'+name] = ref.Y[:,j]
    path = tmp_path / 'reference.csv'
    df.to_csv(path,index=False)
    loaded = TrainingReference.from_csv(path, CLASSES)
    assert np.array_equal(loaded.Y, ref.Y)
    report = audit_predictions(['CCCCCOC(C)=O'], np.array([[.92,.8,.48,.1]]),
                               CLASSES, loaded, options=_opts())
    report.save(tmp_path/'results')
    assert (tmp_path/'results'/'audit_all_label_decisions.csv').is_file()
    assert (tmp_path/'results'/'audit_run_parameters.json').is_file()
    import json
    settings = json.loads((tmp_path/'results'/'audit_run_parameters.json').read_text())
    assert settings['arm_algorithm'].startswith('FP-Growth')
    assert settings['audit_options']['min_support_fraction'] == 0
    assert len(pd.read_csv(tmp_path/'results'/'audit_review_suggestions.csv')) >= 1


def test_predict_can_export_per_class_probabilities_for_audit(monkeypatch):
    # The CI environment may not include the optional mol2vec dependency;
    # this test exercises the probability-export path without computing descriptors.
    import sys, types, importlib.util
    if importlib.util.find_spec('mol2vec') is None:
        fake_pkg = types.ModuleType('mol2vec')
        fake_pkg.__path__ = []
        fake_features = types.ModuleType('mol2vec.features')
        fake_features.mol2alt_sentence = lambda *args, **kw: []
        monkeypatch.setitem(sys.modules, 'mol2vec', fake_pkg)
        monkeypatch.setitem(sys.modules, 'mol2vec.features', fake_features)
    from odor_classifier import predict as predmod

    class FeatureDummy:
        kept_after_variance_ = []
        config = None
        def transform(self, raw):
            return raw

    class ModelDummy:
        def predict_proba(self, values):
            return np.array([[.8,.2,.61,.03]])

    class ADDummy:
        cutoff_ = .75
        def mean_distance(self, values):
            return np.array([.9])

    monkeypatch.setattr(predmod, 'compute_raw_features', lambda smiles,config: pd.DataFrame({'x':[1.]}))
    result = predmod.predict_smiles(['CCO'],ModelDummy(),FeatureDummy(),CLASSES,ADDummy(),
                                  include_audit_data=True)
    assert result['predicted_labels'].iloc[0] == 'fruity, sweet'
    assert result['final_label'].iloc[0] == 'UC'  # raw guesses still available outside AD
    assert result['proba__floral'].iloc[0] == pytest.approx(.2)
    assert result['ad_mean_distance'].iloc[0] == pytest.approx(.9)
    assert result['in_domain'].iloc[0] == False
    report, transformed = predmod.predict_smiles(
        ['CCO'], ModelDummy(), FeatureDummy(), CLASSES, ADDummy(),
        include_audit_data=True, return_transformed_features=True)
    assert list(transformed.columns) == ['x']
    assert transformed.iloc[0, 0] == pytest.approx(1.0)
    assert report.equals(result)
