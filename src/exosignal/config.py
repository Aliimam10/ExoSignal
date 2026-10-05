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

