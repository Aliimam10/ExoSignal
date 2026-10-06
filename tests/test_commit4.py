import numpy as np
import pandas as pd

from exosignal.config import BenchmarkConfig
from exosignal.catalogue import reveal_catalogue_matches
from exosignal.benchmark import _period_match, benchmark_manifest
from exosignal.features import MODEL_FEATURES, feature_row
from exosignal.injection import recovery_kind
from exosignal.ml import split_by_tic, train_models


def test_feature_row_uses_measurements_not_vetting_statuses():
    candidate = {"candidate_id": "C01", "period_days": 3.0, "duration_hours": 2.0, "depth": .01, "depth_err": .001, "bls_snr": 12, "observed_transits": 4}
    dossier = {
        "individual_transit_consistency": {"status": "FAIL", "values": {"fractional_depth_scatter": .2, "weak_or_absent_fraction": .3}},
        "odd_even": {"status": "PASS", "values": {"depth_difference": .001, "difference_sigma": 1}},
        "secondary_eclipse": {"status": "WARNING", "values": {"secondary_depth": .002, "secondary_sigma": 4}},
        "sector_consistency": {"values": {"max_depth_difference": .003, "difference_sigma": 2}},
        "out_of_transit_variability": {"values": {"out_of_transit_rms": .004}},
        "crowding": {"values": {"crowdsap": .9}},
        "pixel_source_check": {"values": {"sector_results": []}},
    }
    row = feature_row(123, candidate, dossier)
    assert set(MODEL_FEATURES).issubset(row)
    assert row["pixel_available"] == 0
    assert np.isnan(row["pixel_offset_pixels"])
    assert "PASS" not in row.values() and "FAIL" not in row.values()


def test_grouped_split_keeps_each_tic_together():
    rows = []
    for label in (0, 1):
        for number in range(6):
            for candidate in range(2):
                row = {"tic_id": label * 100 + number, "label": label, "candidate_id": f"C{candidate}"}
                row.update({name: float(label + candidate + 1) for name in MODEL_FEATURES})
                rows.append(row)
    train, validation, test = split_by_tic(pd.DataFrame(rows), BenchmarkConfig())
    assert not (set(train.tic_id) & set(validation.tic_id))
    assert not (set(train.tic_id) & set(test.tic_id))
    assert not (set(validation.tic_id) & set(test.tic_id))


def test_models_fit_with_explicit_missing_pixel_values(tmp_path):
    rows = []
    for label in (0, 1):
        for number in range(6):
            row = {"tic_id": label * 100 + number, "label": label, "candidate_id": "C01"}
            row.update({name: float(label + 1 + number / 100) for name in MODEL_FEATURES})
            row["pixel_offset_pixels"] = np.nan if number % 2 else float(label)
            rows.append(row)
    result = train_models(pd.DataFrame(rows), tmp_path, BenchmarkConfig(random_forest_trees=10, minimum_calibration_per_class=1))
    assert (tmp_path / "calibrated_random_forest.joblib").exists()
    assert "pr_auc" in result["calibrated_random_forest_test"]


def test_recovery_kind_reports_harmonics_separately():
    assert recovery_kind(4.0, 4.01) == "exact_period"
    assert recovery_kind(4.0, 2.0) == "simple_harmonic"
    assert recovery_kind(4.0, None) == "not_recovered"


def test_catalogue_reveal_requires_period_not_tic_alone():
    ranking = pd.DataFrame({"tic_id": [1, 1], "candidate_id": ["C01", "C02"], "period_days": [3.0, 7.0], "epoch_btjd": [100.0, 100.0]})
    catalogue = pd.DataFrame({"TIC ID": [1], "TOI": [1.01], "TFOPWG Disposition": ["KP"], "Period (days)": [3.0], "Epoch (BJD)": [2457100.0]})
    result = reveal_catalogue_matches(ranking, catalogue)
    assert result.loc[0, "period_match"] == "exact_period"
    assert result.loc[1, "catalogue_reveal"] == "POTENTIALLY UNCATALOGUED TRANSIT-LIKE SIGNAL"


def test_period_match_labels_harmonics_with_same_fractional_tolerance():
    assert _period_match(10.05, 10.0)[0] == "exact"
    assert _period_match(5.025, 10.0)[0] == "half_period"
    assert _period_match(20.1, 10.0)[0] == "double_period"
    assert _period_match(5.2, 10.0) is None


def test_manifest_is_seeded_and_not_short_period_head_selection():
    rows = []
    for label, disposition in ((0, "FP"), (1, "KP")):
        for number in range(20):
            rows.append({"TIC ID": 1000 * (label + 1) + number, "TOI": number, "TFOPWG Disposition": disposition, "Period (days)": 0.5 * 2 ** (number / 3)})
    catalogue = pd.DataFrame(rows)
    first = benchmark_manifest(catalogue, 8)
    second = benchmark_manifest(catalogue, 8)
    assert first.equals(second)
    assert first.groupby("label")["catalogue_period_days"].max().min() > 5
