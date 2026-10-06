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

## Commit 3 vetting

Use an existing target run to create evidence-preserving astrophysical-vetting
dossiers:

```bash
exosignal vet "TIC 402026209"
```

Each diagnostic stores its measured values, explicit criterion, and a short
explanation alongside a `PASS`, `WARNING`, `FAIL`, or `NOT_AVAILABLE` status.
The checks cover individual-event consistency, odd/even depths, a phase-0.5
secondary eclipse, sector depth consistency, out-of-transit RMS, and SPOC
`CROWDSAP` contamination information.

Where a suitable SPOC Target Pixel File is available, ExoSignal computes
in-/out-of-transit and difference images plus a simple flux-loss centroid.
Target Pixel Files are cached per TIC/sector. A centroid consistent with the
target does **not** confirm a planet, and unavailable TPF data are reported as
`Pixel source check: NOT AVAILABLE`, never as a candidate failure.

## Commit 4 benchmark, ranking, discovery, and sensitivity

Commit 4 adds an auditable research workflow, not a planet classifier that
can be trusted without human review. Public ExoFOP dispositions are used only
to create a separate ground-truth manifest. The model features are solely
numeric quantities measured by ExoSignal: BLS/TLS signal strength, depth and
duration, event scatter, odd/even and secondary measurements, sector
consistency, out-of-transit RMS, CROWDSAP, and difference-image offset. The
`PASS`/`WARNING`/`FAIL` strings and every catalogue column are deliberately
excluded. Missing pixel values remain NaN, with an explicit `pixel_available`
feature and median imputation inside each fitted model pipeline.

```bash
# Labels are stored separately and never become model columns.
exosignal label-manifest --per-class 100
exosignal benchmark outputs/benchmark/label_manifest.csv
exosignal train outputs/benchmark/benchmark_features.csv

# This is catalogue-blind and freezes its ranking before reveal.
exosignal discover --sector 2 --maximum-targets 20 \
  --model outputs/models/logistic_regression.joblib
exosignal reveal outputs/discovery/sector_002/pre_crossmatch_ranking.csv

# Inject before long-timescale detrending and rerun BLS.
exosignal inject "TIC 402026209"
```

Train/validation/test partitions are made by TIC (approximately 70/15/15),
so neither different candidates nor sectors from one target cross partitions.
The final test set is not used to fit the models or Random Forest's sigmoid
probability calibration; calibration uses validation TICs only. Reports give
precision, recall, PR-AUC, ROC-AUC, Brier score, confusion matrix, and a
calibration curve, rather than foregrounding accuracy.

Discovery records its MAST sector selection, product choice, deterministic
ordering, attempted-TIC manifest, failures, and an immutable
`pre_crossmatch_ranking.csv`. Catalogue reveal subsequently requires TIC plus
period consistency (or an explicitly reported P/2 or 2P harmonic), rather
than TIC identity alone. An unmatched signal is labelled only
`POTENTIALLY UNCATALOGUED TRANSIT-LIKE SIGNAL`.

When the model directory contains its persisted benchmark split, discovery
automatically excludes every benchmark TIC before it samples the MAST sector;
this prevents a training or held-out benchmark target being reused as the
headline blind-discovery population.

The compact injection grid writes exact-period and harmonic recoveries
separately. It injects into saved per-sector PDCSAP-normalised data before the
long-timescale detrending and BLS stages; quality removal and per-sector scale
normalisation have already occurred. It is a sensitivity experiment, not a
completeness claim.
