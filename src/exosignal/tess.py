"""Public MAST/TESS retrieval limited to SPOC PDCSAP light-curve products."""

from __future__ import annotations

import re
from typing import Any

import lightkurve as lk
import numpy as np

from .errors import ExoSignalError, InvalidTicError, NoTessDataError, UnusableLightCurveError


def parse_tic_id(value: str | int) -> int:
    """Accept ``307210830`` or the user-friendly ``TIC 307210830`` form."""
    match = re.fullmatch(r"\s*(?:TIC\s*)?(\d+)\s*", str(value), flags=re.IGNORECASE)
    if not match:
        raise InvalidTicError(f"'{value}' is not a valid TIC ID.")
    return int(match.group(1))


def discover_spoc_products(tic_id: int):
    """Query MAST for downloadable SPOC TESS light-curve products."""
    try:
        result = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
    except Exception as error:
        raise ExoSignalError(
            "Could not query MAST for TESS products. Check network access and try again."
        ) from error
    if len(result) == 0:
        raise NoTessDataError(f"No SPOC TESS light curves were found for TIC {tic_id}.")
    return result


def _sector_from_lightcurve(lightcurve: Any, fallback: int | None = None) -> int:
    value = lightcurve.meta.get("SECTOR", fallback)
    if value is None:
        raise UnusableLightCurveError("A downloaded light curve has no sector identifier.")
    return int(value)


def _column_values(lightcurve: Any, column: str, required: bool = False) -> np.ndarray | None:
    if column not in lightcurve.colnames:
        if required:
            raise UnusableLightCurveError(f"Product lacks required {column} column.")
        return None
    values = lightcurve[column]
    return np.asarray(getattr(values, "value", values))


def selected_product_indices(search_result) -> list[int]:
    """Choose one SPOC light curve per sector, preferring 120-second cadence.

    Some targets have both fast and standard cadence products in the same
    sector. Combining them would duplicate the same observations and give that
    sector disproportionate weight, so Commit 1 uses the conventional SPOC
    two-minute product where it exists.
    """
    per_sector: dict[int, list[int]] = {}
    for index, row in enumerate(search_result.table):
        per_sector.setdefault(int(row["sequence_number"]), []).append(index)

    def preference(index: int) -> tuple[float, float, int]:
        exposure = float(search_result.table[index]["exptime"])
        return (abs(exposure - 120.0), exposure, index)

    return [min(indices, key=preference) for _, indices in sorted(per_sector.items())]


def download_spoc_lightcurves(search_result, download_dir: str | None = None) -> tuple[list[tuple[int, Any]], list[dict[str, str]]]:
    """Download products one at a time so one failed sector does not abort a TIC.

    ``download_dir`` keeps the MAST cache alongside the reproducible target
    run rather than in a hidden user-level cache.
    """
    products: list[tuple[int, Any]] = []
    failures: list[dict[str, str]] = []
    for index in selected_product_indices(search_result):
        try:
            lightcurve = search_result[index].download(
                download_dir=download_dir, quality_bitmask=None
            )
            if lightcurve is None:
                raise NoTessDataError("MAST returned no light curve for this product.")
            # Explicitly require PDCSAP, never silently fall back to SAP flux.
            _column_values(lightcurve, "pdcsap_flux", required=True)
            products.append((_sector_from_lightcurve(lightcurve), lightcurve))
        except Exception as error:
            failures.append({"product_index": str(index), "reason": str(error)})
    if not products:
        raise ExoSignalError(
            "MAST listed products but none could be downloaded. Check network access and try again."
        )
    return products, failures


def lightcurve_arrays(lightcurve: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Extract numerical arrays from a Lightkurve object without changing values."""
    time = np.asarray(lightcurve.time.value)
    flux = _column_values(lightcurve, "pdcsap_flux", required=True)
    errors = _column_values(lightcurve, "pdcsap_flux_err")
    quality = _column_values(lightcurve, "quality")
    return time, flux, errors, quality


def target_metadata(tic_id: int, products: list[tuple[int, Any]]) -> dict[str, object]:
    """Return useful header metadata, with no catalogue classification fields."""
    first = products[0][1]
    meta = first.meta
    fields = {
        "ra_deg": ("RA_OBJ", "RA"),
        "dec_deg": ("DEC_OBJ", "DEC"),
        "tmag": ("TESSMAG", "TMAG"),
        "stellar_radius_rsun": ("RADIUS", "RADIUS_STAR"),
        "teff_k": ("TEFF",),
        "logg": ("LOGG",),
        "crowdsap": ("CROWDSAP",),
    }
    output: dict[str, object] = {"tic_id": tic_id}
    for output_name, keys in fields.items():
        for key in keys:
            value = meta.get(key)
            if value is not None:
                output[output_name] = _json_value(value)
                break
    output["sectors_available"] = sorted({sector for sector, _ in products})
    output["metadata_source"] = "SPOC light-curve product headers"
    return output


def _json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value
