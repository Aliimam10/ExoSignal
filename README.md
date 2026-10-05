# ExoSignal

ExoSignal is a compact, scientifically careful TESS transit-search and
candidate-vetting project. This first implementation stage retrieves public
SPOC PDCSAP light curves for a TIC, preprocesses each available sector
independently, and writes a local, inspectable data product.

It does not yet search for transits, fit transit models, classify candidates,
or claim planetary discoveries.

## Run the data pipeline

Create an environment, install the project, then run:

```bash
python -m pip install -e '.[dev]'
exosignal analyze "TIC 307210830"
```

The command writes `outputs/targets/307210830/` by default. It contains the
processed cadence table, sector diagnostics, target metadata, run
configuration, and a static PNG light-curve plot.

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

This repository currently implements **Commit 1 only**: TESS data retrieval,
sector-aware preprocessing, persistence, plotting, and tests.
