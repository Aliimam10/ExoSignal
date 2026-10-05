"""Explicit, documented configuration for Commit 1 processing."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PreprocessingConfig:
    """Parameters chosen to preserve plausible transit-scale flux dips.

    ``detrend_window_days`` is longer than the maximum 10-hour transit duration
    planned for ExoSignal's initial search. A later transit-aware second pass
    can supply a mask without changing this public processing interface.
    """

    quality_mode: str = "zero_only"
    positive_outlier_sigma: float = 8.0
    detrend_window_days: float = 3.0
    gap_break_days: float = 0.5
    min_segment_points: int = 21

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SearchConfig:
    """Explicit first-pass search choices, to be revisited with validation."""

    min_period_days: float = 0.5
    max_period_days: float = 30.0
    min_duration_hours: float = 0.5
    max_duration_hours: float = 10.0
    minimum_transits: int = 3
    minimum_snr: float = 6.0
    max_candidates: int = 3
    transit_mask_duration_factor: float = 1.5
    max_bls_period_samples: int = 50_000
    bls_oversample: int = 3
    bls_bin_minutes: float = 10.0
    tls_bin_minutes: float = 30.0
    tls_period_window_fraction: float = 0.001
    tls_threads: int = 1
    tls_enabled: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VettingConfig:
    """Conservative, explicit criteria for Commit 3 candidate diagnostics."""

    event_detection_snr: float = 3.0
    minimum_event_inner_points: int = 2
    minimum_event_outer_points: int = 4
    odd_even_warning_sigma: float = 3.0
    odd_even_fail_sigma: float = 5.0
    secondary_warning_sigma: float = 3.0
    secondary_fail_sigma: float = 5.0
    sector_warning_sigma: float = 3.0
    sector_fail_sigma: float = 5.0
    weak_event_warning_fraction: float = 0.35
    variability_warning_rms: float = 0.01
    variability_fail_rms: float = 0.03
    crowding_warning_minimum: float = 0.8
    crowding_fail_minimum: float = 0.5
    pixel_consistent_offset_pixels: float = 0.5
    pixel_off_target_offset_pixels: float = 1.0
    pixel_minimum_in_cadences: int = 3
    pixel_minimum_out_cadences: int = 5
    generate_pixel_plots: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Fixed, deliberately small modelling choices for Commit 4."""

    random_seed: int = 20261005
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    random_forest_trees: int = 300
    probability_threshold: float = 0.5
    calibration_method: str = "sigmoid"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
