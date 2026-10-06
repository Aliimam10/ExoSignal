"""The explicit Commit 1 target workflow: retrieve, process, save, plot."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import PreprocessingConfig, SearchConfig, VettingConfig
from .dossier import run_vetting_dossiers
from .errors import ExoSignalError
from .plotting import (
    individual_transit_data,
    plot_bls_periodogram,
    plot_individual_transits,
    plot_phase_folded,
    plot_processed_lightcurve,
)
from .preprocessing import combine_sectors, process_sector
from .search import BlsResult, add_stellar_radius_interpretation, search_transits
from .tess import (
    discover_spoc_products,
    download_spoc_lightcurves,
    lightcurve_arrays,
    parse_tic_id,
    target_metadata,
)


def analyse_target(
    tic: str | int,
    output_root: Path | str = "outputs/targets",
    config: PreprocessingConfig | None = None,
    search_config: SearchConfig | None = None,
    run_vetting: bool = True,
    vetting_config: VettingConfig | None = None,
    generate_plots: bool = True,
) -> dict[str, object]:
    """Run Commit 2's retrieval, processing, and transit-like signal search.

    A bad product is recorded and skipped; the target succeeds if at least one
    sector remains usable. This makes future multi-target runs resilient.
    """
    tic_id = parse_tic_id(tic)
    config = config or PreprocessingConfig()
    search_config = search_config or SearchConfig()
    run_directory = Path(output_root) / str(tic_id)
    run_directory.mkdir(parents=True, exist_ok=True)

    search_result = discover_spoc_products(tic_id)
    products, download_failures = download_spoc_lightcurves(
        search_result, download_dir=str(run_directory / "mast_cache")
    )
    metadata = target_metadata(tic_id, products)
    processed = []
    skipped: list[dict[str, object]] = list(download_failures)
    for sector, lightcurve in products:
        try:
            time, flux, errors, quality = lightcurve_arrays(lightcurve)
            processed.append(
                process_sector(
                    sector=sector,
                    time=time,
                    flux=flux,
                    flux_err=errors,
                    quality=quality,
                    config=config,
                )
            )
        except (ExoSignalError, ValueError) as error:
            skipped.append({"sector": sector, "reason": str(error)})

    combined = combine_sectors(processed)
    if combined.empty:
        details = "; ".join(str(item["reason"]) for item in skipped) or "no usable products"
        raise ExoSignalError(f"TIC {tic_id} has no usable processed sectors: {details}")

    initial_data_path = run_directory / "processed_lightcurve_initial.csv"
    data_path = run_directory / "processed_lightcurve.csv"
    sector_path = run_directory / "sectors.json"
    metadata_path = run_directory / "target_metadata.json"
    config_path = run_directory / "preprocessing_config.json"
    search_config_path = run_directory / "search_config.json"
    plot_path = run_directory / "processed_lightcurve.png"
    candidates, refined, periodograms = search_transits(combined, config, search_config)
    candidates = [
        add_stellar_radius_interpretation(candidate, metadata.get("stellar_radius_rsun"))
        for candidate in candidates
    ]
    periodograms = [
        BlsResult(candidate=candidate, periods=result.periods, power=result.power)
        for candidate, result in zip(candidates, periodograms)
    ]
    combined.to_csv(initial_data_path, index=False)
    refined.to_csv(data_path, index=False)
    _write_json(metadata_path, metadata)
    _write_json(
        sector_path,
        {
            "processed_sectors": [item.diagnostics for item in processed],
            "skipped_products": skipped,
        },
    )
    _write_json(config_path, config.to_dict())
    _write_json(search_config_path, search_config.to_dict())
    if generate_plots:
        plot_processed_lightcurve(refined, plot_path, tic_id)
    candidate_directory = run_directory / "candidates"
    candidate_directory.mkdir(exist_ok=True)
    # A rerun for the same target must not leave dossiers from a previous,
    # differently configured search beside the current candidate table.
    for pattern in (
        "c??_*.png", "c??_*.csv", "c??_*.json", "candidates.json", "candidates.csv"
    ):
        for stale_output in candidate_directory.glob(pattern):
            stale_output.unlink()
    candidate_dicts = [candidate.to_dict() for candidate in candidates]
    _write_json(candidate_directory / "candidates.json", candidate_dicts)
    pd.DataFrame(candidate_dicts).to_csv(candidate_directory / "candidates.csv", index=False)
    candidate_outputs: list[str] = []
    for candidate, periodogram in zip(candidates, periodograms):
        stem = candidate.candidate_id.lower()
        periodogram_path = candidate_directory / f"{stem}_bls_periodogram.png"
        phase_path = candidate_directory / f"{stem}_phase_folded.png"
        transit_data_path = candidate_directory / f"{stem}_individual_transits.csv"
        transit_plot_path = candidate_directory / f"{stem}_individual_transits.png"
        if generate_plots:
            plot_bls_periodogram(periodogram, periodogram_path)
            plot_phase_folded(refined, candidate, phase_path)
        individual_transit_data(refined, candidate).to_csv(transit_data_path, index=False)
        if generate_plots:
            plot_individual_transits(refined, candidate, transit_plot_path)
            candidate_outputs.extend([str(periodogram_path), str(phase_path), str(transit_data_path), str(transit_plot_path)])
        else:
            candidate_outputs.append(str(transit_data_path))
    vetting_summary = (
        run_vetting_dossiers(tic_id, output_root, vetting_config)
        if run_vetting
        else {"tic_id": tic_id, "skipped": True}
    )
    return {
        "tic_id": tic_id,
        "run_directory": str(run_directory),
        "processed_sectors": [item.sector for item in processed],
        "skipped_products": skipped,
        "cadences": len(refined),
        "candidates": candidate_dicts,
        "vetting": vetting_summary,
        "outputs": [
            str(initial_data_path), str(data_path), str(sector_path), str(metadata_path),
            str(config_path), str(search_config_path), str(plot_path),
            str(candidate_directory / "candidates.json"), str(candidate_directory / "candidates.csv"),
            *candidate_outputs,
        ],
    }


def _write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
