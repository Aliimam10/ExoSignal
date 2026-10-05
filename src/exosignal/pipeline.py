"""The explicit Commit 1 target workflow: retrieve, process, save, plot."""

from __future__ import annotations

import json
from pathlib import Path

from .config import PreprocessingConfig
from .errors import ExoSignalError
from .plotting import plot_processed_lightcurve
from .preprocessing import combine_sectors, process_sector
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
) -> dict[str, object]:
    """Run Commit 1's complete public-data pipeline for one TIC.

    A bad product is recorded and skipped; the target succeeds if at least one
    sector remains usable. This makes future multi-target runs resilient.
    """
    tic_id = parse_tic_id(tic)
    config = config or PreprocessingConfig()
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

    data_path = run_directory / "processed_lightcurve.csv"
    sector_path = run_directory / "sectors.json"
    metadata_path = run_directory / "target_metadata.json"
    config_path = run_directory / "preprocessing_config.json"
    plot_path = run_directory / "processed_lightcurve.png"
    combined.to_csv(data_path, index=False)
    _write_json(metadata_path, metadata)
    _write_json(
        sector_path,
        {
            "processed_sectors": [item.diagnostics for item in processed],
            "skipped_products": skipped,
        },
    )
    _write_json(config_path, config.to_dict())
    plot_processed_lightcurve(combined, plot_path, tic_id)
    return {
        "tic_id": tic_id,
        "run_directory": str(run_directory),
        "processed_sectors": [item.sector for item in processed],
        "skipped_products": skipped,
        "cadences": len(combined),
        "outputs": [str(data_path), str(sector_path), str(metadata_path), str(config_path), str(plot_path)],
    }


def _write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
