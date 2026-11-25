"""Synthetic regression tests for the declared odor dictionary vs generated targets."""
import pandas as pd
import pytest

from odor_classifier.target_labels import (
    build_binary_targets, dictionary_index, dictionary_target_audit,
)


def _dictionary():
    return pd.DataFrame({
        "class": ["fruity", "floral", "marine", "UC"],
        "type1": ["apple", "rose", "sea", None],
        "type2": ["pear", "jasmine", None, None],
    })


def test_all_real_classes_preserved_even_when_zero_positive():
    compounds = pd.DataFrame({
        "odor1": ["apple", "rose"],
        "odor2": [None, "jasmine"],
        "class1": ["wrong", "wrong"],
    })
    Y, primary, report = build_binary_targets(compounds, _dictionary())
    assert list(Y.columns) == ["fruity", "floral", "marine"]
    assert Y["marine"].sum() == 0
    assert Y["fruity"].tolist() == [1, 0]
    assert Y["floral"].tolist() == [0, 1]
    assert primary.tolist() == ["fruity", "floral"]
    assert report.empty
    audit = dictionary_target_audit(_dictionary(), Y).set_index("Dictionary class")
    assert audit.loc["marine", "Status"] == "Present (no positive compounds)"
    assert audit.loc["UC", "Status"].startswith("Excluded: UC")
    assert audit.loc["fruity", "Positive compounds"] == 1


def test_nonreserved_label_missing_from_export_is_detected():
    Y, _, _ = build_binary_targets(pd.DataFrame({"odor1": ["apple"]}), _dictionary())
    audit = dictionary_target_audit(_dictionary(), Y.drop(columns=["marine"]))
    missing = audit[audit["Status"].str.contains("MISSING")]
    assert missing["Dictionary class"].tolist() == ["marine"]


def test_conflicting_odor_to_class_mapping_fails():
    d = _dictionary()
    d.loc[d["class"] == "floral", "type1"] = "apple"
    with pytest.raises(ValueError, match="Dictionary ambiguity"):
        dictionary_index(d)


def test_supplied_dataset_has_22_targets_and_consistent_class_annotations():
    from odor_classifier.config import DEFAULT_CONFIG
    from odor_classifier.io_utils import load_compounds, load_class_dictionary
    from odor_classifier.target_labels import normalize_term

    compounds = load_compounds(config=DEFAULT_CONFIG)
    dictionary = load_class_dictionary(config=DEFAULT_CONFIG)
    classes, mapping = dictionary_index(dictionary)
    Y, primary, report = build_binary_targets(compounds, dictionary)
    assert Y.shape == (5288, 22)
    assert len(dictionary) == len(classes) == 22
    assert report.empty and primary.notna().all()
    assert Y.sum(axis=1).gt(0).all()
    for i in range(1, 11):
        expected = compounds[f"odor{i}"].map(lambda value: mapping.get(normalize_term(value)))
        assert expected.fillna("").eq(compounds[f"class{i}"].fillna("")).all()
