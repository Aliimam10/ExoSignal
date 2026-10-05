# ExoSignal

ExoSignal is a compact, scientifically careful TESS transit-search and
candidate-vetting project. This first implementation stage retrieves public
SPOC PDCSAP light curves for a TIC, preprocesses each available sector
independently, and writes a local, inspectable data product.

It uses Box Least Squares (BLS) to find periodic transit-like signals, then
optionally refines promising peaks with Transit Least Squares (TLS). It does
not classify candidates or claim planetary discoveries.

## Run the data pipeline

Create an environment, install the project, then run:

```bash
python -m pip install -e '.[dev]'
exosignal analyze "TIC 307210830"
```

The command writes `outputs/targets/307210830/` by default. It contains the
processed cadence table, sector diagnostics, target metadata, run
configuration, BLS candidate outputs, and static diagnostic PNGs.

## Commit 1 preprocessing choices

Each SPOC sector is handled separately. Cadences with non-finite time or
PDCSAP flux are removed, as are non-zero `QUALITY` cadences. The flux is
normalised by its sector median. Only extreme *positive* isolated excursions
are removed (8 robust standard deviations); low-flux points are retained so a
possible transit is not discarded as an outlier.

Long-timescale variability is removed with a 3-day running-median trend, run
separately on continuous pieces of a sector (a gap of 0.5 day begins a new
piece). The window is deliberately much longer than the 0.5--10 hour transit
durations intended for the eventual search. This is a first-pass detrend only:
future transit masking may be passed in through the preprocessing interface,
but is not implemented until transit detection exists.

When MAST offers both fast and standard SPOC products for a sector, the
pipeline selects the standard 120-second product. This prevents duplicated
observations of the same sector from receiving excess weight. The source
products are cached inside the target run for provenance.

The output `flux` is relative detrended flux and `flux_err` is propagated as
the PDCSAP uncertainty divided by the local trend. Original per-sector median
normalisation is retained in `raw_normalized_flux` for inspection.

## Scope

## Commit 2 search

BLS searches 0.5--30 day periods (also limited to one third of the observed
baseline) and 0.5--10 hour durations. It requires at least three observed
transit windows and a BLS SNR of 6. It returns only the strongest independent
peak per iteration, rejecting near-duplicates and simple P/2, 2P, and 3P
aliases. Up to three signals are searched for: each accepted signal's transit
windows are excluded for the next residual search.

After preliminary BLS detection, ExoSignal recalculates each sector's trend
with a 1.5-duration-wide predicted transit mask. The measurements themselves
are kept; only the baseline estimator sees interpolated values at those times.
The candidate is then remeasured locally with BLS. TLS is a second-stage
refinement only and may be skipped with `--skip-tls`.

This repository currently implements **Commit 2 only**: TESS retrieval,
sector-aware preprocessing, BLS/TLS candidate measurement, static candidate
plots, individual-transit extraction, persistence, and tests.
