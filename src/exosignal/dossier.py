"""Persist Commit 3 vetting dossiers independently from search execution."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import VettingConfig
from .plotting import plot_pixel_diagnostic
from .search import Candidate
from .vetting import Diagnostic, candidate_vetting, pixel_source_diagnostic, retrieve_spoc_tpfs


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run_vetting_dossiers(tic_id: int, output_root: Path | str = "outputs/targets", config: VettingConfig | None = None) -> dict[str, object]:
    """Vets retained candidates and writes evidence-rich local dossier files."""
    config = config or VettingConfig()
    run_directory = Path(output_root) / str(tic_id)
    candidate_directory = run_directory / "candidates"
    records = json.loads((candidate_directory / "candidates.json").read_text())
    for record in records:
        # Permit Commit 2 output directories to be vetted without forcing a
        # repeat of their already-expensive BLS/TLS search.
        record.setdefault("radius_ratio", max(float(record["depth"]), 0.0) ** 0.5)
        record.setdefault("approximate_planet_radius_earth", None)
    candidates = [Candidate(**record) for record in records]
    data = pd.read_csv(run_directory / "processed_lightcurve.csv")
    metadata = json.loads((run_directory / "target_metadata.json").read_text())
    _write_json(run_directory / "vetting_config.json", config.to_dict())

    if config.retrieve_pixel_data:
        tpfs, retrieval_errors = retrieve_spoc_tpfs(
            tic_id, set(data["sector"].astype(int).unique()), run_directory / "mast_cache" / "tpf",
            config.pixel_download_timeout_seconds,
        )
    else:
        tpfs, retrieval_errors = {}, ["Pixel retrieval intentionally disabled for this scalable benchmark run."]
    summaries: list[dict[str, object]] = []
    for candidate in candidates:
        if config.retrieve_pixel_data:
            pixel_diagnostic, pixel_measurements = pixel_source_diagnostic(candidate, tpfs, config)
        else:
            pixel_diagnostic, pixel_measurements = (
                Diagnostic("NOT_AVAILABLE", {"sector_results": []}, "Pixel retrieval disabled for this benchmark configuration.", "Pixel source check: NOT AVAILABLE."),
                [],
            )
        if retrieval_errors:
            pixel_diagnostic.values["retrieval_errors"] = retrieval_errors
        diagnostics, events = candidate_vetting(data, metadata, candidate, config, pixel_diagnostic)
        stem = candidate.candidate_id.lower()
        _write_json(candidate_directory / f"{stem}_vetting.json", {name: value.to_dict() for name, value in diagnostics.items()})
        events.to_csv(candidate_directory / f"{stem}_event_metrics.csv", index=False)
        _write_json(candidate_directory / f"{stem}_pixel_source.json", pixel_diagnostic.to_dict())
        if config.generate_pixel_plots:
            usable = next((item for item in pixel_measurements if item.out_image is not None), None)
            if usable is not None:
                plot_pixel_diagnostic(usable, candidate_directory / f"{stem}_pixel_diagnostic.png", candidate)
        summaries.append({"candidate_id": candidate.candidate_id, "overall": diagnostics["overall"].status, "pixel": pixel_diagnostic.status})
    return {"tic_id": tic_id, "candidates": summaries, "tpf_sectors_retrieved": sorted(tpfs), "tpf_retrieval_errors": retrieval_errors}
