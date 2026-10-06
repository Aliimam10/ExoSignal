"""Public catalogue labels and post-hoc signal-consistency matching only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXOFOP_TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=pipe"


def download_toi_catalogue(cache_path: Path | str) -> pd.DataFrame:
    """Cache public ExoFOP TOI data; never expose its fields to model inputs."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_csv(path, sep="|")
    table = pd.read_csv(EXOFOP_TOI_URL, sep="|", comment="#")
    table.to_csv(path, sep="|", index=False)
    return table


def reliable_label_manifest(catalogue: pd.DataFrame, maximum_per_class: int | None = None) -> pd.DataFrame:
    """Select only KP/CP positives and FP/FA negatives for benchmark labels.

    Unresolved TOIs are excluded.  This returned table is a label manifest,
    not a feature table, and contains no columns consumed by the classifier.
    """
    disposition = catalogue["TFOPWG Disposition"].astype(str).str.upper().str.strip()
    positive = catalogue.loc[disposition.isin(["KP", "CP"])].copy()
    negative = catalogue.loc[disposition.isin(["FP", "FA"])].copy()
    selected = []
    for label, frame in ((1, positive), (0, negative)):
        frame = frame.dropna(subset=["TIC ID"]).sort_values(["TIC ID", "TOI"])
        frame = frame.drop_duplicates("TIC ID")
        if maximum_per_class is not None:
            frame = frame.head(maximum_per_class)
        selected.append(pd.DataFrame({"tic_id": frame["TIC ID"].astype(int), "label": label, "label_source": "ExoFOP TFOPWG disposition", "catalogue_disposition": disposition.loc[frame.index].to_numpy()}))
    return pd.concat(selected, ignore_index=True).sort_values(["label", "tic_id"], ignore_index=True)


def _period_match(candidate_period: float, catalogue_period: float, tolerance: float) -> str | None:
    if not np.isfinite(candidate_period) or not np.isfinite(catalogue_period) or catalogue_period <= 0:
        return None
    ratio = candidate_period / catalogue_period
    if abs(ratio - 1.0) <= tolerance:
        return "exact_period"
    if abs(ratio - 0.5) <= tolerance or abs(ratio - 2.0) <= 2 * tolerance:
        return "simple_harmonic"
    return None


def reveal_catalogue_matches(ranking: pd.DataFrame, catalogue: pd.DataFrame, period_tolerance: float = 0.01) -> pd.DataFrame:
    """Match frozen pre-crossmatch signals by TIC *and* period/harmonic relation."""
    output = ranking.copy()
    revealed: list[dict[str, Any]] = []
    for _, candidate in output.iterrows():
        rows = catalogue.loc[catalogue["TIC ID"].astype("Int64") == int(candidate["tic_id"])]
        matches = []
        for _, row in rows.iterrows():
            kind = _period_match(float(candidate["period_days"]), float(row.get("Period (days)", np.nan)), period_tolerance)
            if kind is None:
                continue
            # Epoch is supporting evidence when both are available. ExoFOP BJD
            # becomes BTJD by subtracting 2457000; a missing epoch does not
            # turn a period-consistent signal into a TIC-only match.
            epoch_delta = np.nan
            catalog_epoch = float(row.get("Epoch (BJD)", np.nan)) - 2457000.0
            if np.isfinite(catalog_epoch) and np.isfinite(float(candidate.get("epoch_btjd", np.nan))):
                delta = abs(float(candidate["epoch_btjd"]) - catalog_epoch) % float(row["Period (days)"])
                epoch_delta = min(delta, float(row["Period (days)"]) - delta)
            matches.append({"toi": row.get("TOI"), "catalogue_disposition": row.get("TFOPWG Disposition"), "catalogue_period_days": row.get("Period (days)"), "period_match": kind, "epoch_delta_days": epoch_delta})
        revealed.append(matches[0] if matches else {"catalogue_reveal": "POTENTIALLY UNCATALOGUED TRANSIT-LIKE SIGNAL"})
    reveal_frame = pd.DataFrame(revealed)
    return pd.concat([output.reset_index(drop=True), reveal_frame], axis=1)
