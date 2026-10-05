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
