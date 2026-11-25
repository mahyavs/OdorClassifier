from odor_classifier.config import DEFAULT_CONFIG
from odor_classifier.io_utils import build_odor_to_class_map, load_class_dictionary, load_compounds
from odor_classifier.labeling import extract_labels, known_classes_from_labels, validate_against_dictionary


def test_load_compounds_has_expected_columns():
    df = load_compounds(config=DEFAULT_CONFIG)
    assert "SMILES" in df.columns
    assert "odor1" in df.columns
    assert len(df) > 1000


def test_load_class_dictionary_has_22_rows():
    df = load_class_dictionary(config=DEFAULT_CONFIG)
    assert len(df) == 22
    assert "class" in df.columns


def test_odor_to_class_map_contains_known_words():
    class_dict = load_class_dictionary(config=DEFAULT_CONFIG)
    mapping = build_odor_to_class_map(class_dict)
    assert mapping["fruity"] == "fruity"
    assert mapping["caramellic"] == "sweet"
    assert mapping["sour"] == "sharp"


def test_extract_labels_fully_populated():
    df = load_compounds(config=DEFAULT_CONFIG)
    labels = extract_labels(df, DEFAULT_CONFIG)
    assert labels.notna().all()
    classes = known_classes_from_labels(labels)
    assert 15 <= len(classes) <= 25




def test_uploaded_dictionary_maps_all_compounds_to_22_targets():
    from odor_classifier.target_labels import build_binary_targets
    y, primary, unmapped = build_binary_targets(load_compounds(), load_class_dictionary(), strict_unmapped=True)
    assert y.shape[1] == 22
    assert primary.notna().all()
    assert unmapped.empty
