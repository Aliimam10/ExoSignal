"""Static, inspectable plotting for processed TESS light curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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

