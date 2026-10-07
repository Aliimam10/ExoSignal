import pandas as pd
import numpy as np

from exosignal.catalogue_benchmark import FEATURE_COLUMNS, select_labelled_toi_targets
from exosignal.ml import strong_class_thresholds


def test_catalogue_selection_keeps_nasa_feature_names_and_hides_disposition():
    rows = []
    for label, disposition, start in ((1, "KP", 1000), (0, "FP", 2000)):
        for number in range(10):
            row = {"tid": start + number, "toi": float(start + number) + .01, "tfopwg_disp": disposition}
            row.update({feature: float(number + 1) for feature in FEATURE_COLUMNS})
            row["pl_orbper"] = .5 * 2 ** (number / 2)
            rows.append(row)
    manifest, features = select_labelled_toi_targets(pd.DataFrame(rows), per_class=4)
    assert len(manifest) == len(features) == 8
    assert set(features.columns) == {"tic_id", "toi", *FEATURE_COLUMNS, "label"}
    assert "tfopwg_disp" not in features
    assert manifest.groupby("label")["tic_id"].nunique().to_dict() == {0: 4, 1: 4}


def test_catalogue_selection_can_use_all_eligible_and_exclude_prior_test_tics():
    rows = []
    for label, disposition, start in ((1, "KP", 1000), (0, "FP", 2000)):
        for number in range(4):
            row = {"tid": start + number, "toi": float(start + number) + .01, "tfopwg_disp": disposition}
            row.update({feature: float(number + 1) for feature in FEATURE_COLUMNS})
            rows.append(row)
    manifest, features = select_labelled_toi_targets(pd.DataFrame(rows), per_class=None, excluded_tic_ids={1001, 2002})
    assert len(manifest) == len(features) == 6
    assert {1001, 2002}.isdisjoint(set(features["tic_id"]))


def test_strong_class_bands_are_derived_from_validation_scores():
    thresholds = strong_class_thresholds(
        np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]),
        np.array([.10, .20, .30, .40, .45, .50, .60, .70, .80]),
        minimum_purity=.8,
    )
    assert thresholds["source"] == "validation TICs only"
    assert thresholds["strong_fp_eb_max_score"] == .50
    assert thresholds["strong_planet_like_min_score"] == .60
