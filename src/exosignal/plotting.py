"""Static, inspectable plotting for processed TESS light curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .search import BlsResult, Candidate, candidate_transit_mask


def plot_processed_lightcurve(data: pd.DataFrame, output_path: Path, tic_id: int) -> None:
    """Write a compact sector-coloured static plot without interactive tooling."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    for sector, group in data.groupby("sector", sort=True):
        axis.scatter(group["time"], group["flux"], s=1.2, alpha=0.55, label=f"Sector {sector}")
    axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.55)
    axis.set(
        title=f"TIC {tic_id} — processed SPOC PDCSAP flux",
        xlabel="Time (BTJD)",
        ylabel="Relative flux",
    )
    axis.legend(markerscale=4, ncol=min(4, data["sector"].nunique()), fontsize=8)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def plot_bls_periodogram(result: BlsResult, output_path: Path) -> None:
    """Write a static BLS periodogram with the selected peak marked."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.plot(result.periods, result.power, color="#1f77b4", linewidth=0.8)
    axis.axvline(result.candidate.period_days, color="#d62728", linestyle="--", label="selected")
    axis.set(xlabel="Period (days)", ylabel="BLS SNR", title=f"{result.candidate.candidate_id} BLS periodogram")
    axis.legend()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def phase_folded_data(data: pd.DataFrame, candidate: Candidate) -> pd.DataFrame:
    """Return phase-folded measurements in hours from the candidate epoch."""
    folded_hours = (
        (data["time"].to_numpy() - candidate.epoch_btjd + candidate.period_days / 2)
        % candidate.period_days
        - candidate.period_days / 2
    ) * 24.0
    return pd.DataFrame({"phase_hours": folded_hours, "flux": data["flux"], "flux_err": data["flux_err"]}).sort_values("phase_hours")


def plot_phase_folded(data: pd.DataFrame, candidate: Candidate, output_path: Path) -> None:
    """Write an interpretable static phase-folded transit candidate plot."""
    folded = phase_folded_data(data, candidate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.scatter(folded["phase_hours"], folded["flux"], s=2, alpha=0.35, color="#1f77b4")
    bins = np.linspace(folded["phase_hours"].min(), folded["phase_hours"].max(), 121)
    labels = np.digitize(folded["phase_hours"], bins)
    centres, means = [], []
    for index in range(1, len(bins)):
        values = folded.loc[labels == index, "flux"]
        if len(values):
            centres.append((bins[index - 1] + bins[index]) / 2)
            means.append(values.mean())
    axis.plot(centres, means, color="#d62728", linewidth=1.4, label="binned mean")
    half_duration = candidate.duration_hours / 2
    axis.axvspan(-half_duration, half_duration, color="black", alpha=0.08, label="BLS duration")
    axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    axis.set(
        xlim=(-max(candidate.duration_hours * 3, 3), max(candidate.duration_hours * 3, 3)),
        xlabel="Time from predicted mid-transit (hours)",
        ylabel="Relative flux",
        title=f"{candidate.candidate_id} — P={candidate.period_days:.6f} d",
    )
    axis.legend(fontsize=8)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def individual_transit_data(data: pd.DataFrame, candidate: Candidate) -> pd.DataFrame:
    """Extract labelled local windows around each expected observed transit."""
    time = data["time"].to_numpy()
    event_number = np.rint((time - candidate.epoch_btjd) / candidate.period_days).astype(int)
    midtime = candidate.epoch_btjd + event_number * candidate.period_days
    relative_hours = (time - midtime) * 24.0
    keep = np.abs(relative_hours) <= max(3 * candidate.duration_hours, 3.0)
    return pd.DataFrame(
        {
            "event_number": event_number[keep],
            "predicted_midtime_btjd": midtime[keep],
            "relative_time_hours": relative_hours[keep],
            "flux": data.loc[keep, "flux"].to_numpy(),
            "flux_err": data.loc[keep, "flux_err"].to_numpy(),
        }
    ).sort_values(["event_number", "relative_time_hours"])


def plot_individual_transits(data: pd.DataFrame, candidate: Candidate, output_path: Path) -> None:
    """Write a compact static overview of up to nine individual events."""
    events = individual_transit_data(data, candidate)
    selected = list(events["event_number"].drop_duplicates())[:9]
    columns = 3
    rows = max(1, int(np.ceil(len(selected) / columns)))
    figure, axes = plt.subplots(rows, columns, figsize=(10, 2.6 * rows), squeeze=False, constrained_layout=True)
    for axis, event in zip(axes.flat, selected):
        event_data = events.loc[events["event_number"] == event]
        axis.scatter(event_data["relative_time_hours"], event_data["flux"], s=5, alpha=0.6)
        axis.axvline(0, color="black", linewidth=0.7)
        axis.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        axis.set_title(f"Transit {event}", fontsize=9)
        axis.set_xlabel("Hours", fontsize=8)
        axis.set_ylabel("Rel. flux", fontsize=8)
    for axis in axes.flat[len(selected) :]:
        axis.set_visible(False)
    figure.suptitle(f"{candidate.candidate_id} individual transit windows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
