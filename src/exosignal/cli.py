"""Small command-line entry point for Commit 1."""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path

import pandas as pd

from .errors import ExoSignalError
from .benchmark import benchmark_manifest, build_benchmark, supplemental_negative_manifest
from .catalogue import download_toi_catalogue, reveal_catalogue_matches
from .config import SearchConfig
from .discovery import run_discovery
from .dossier import run_vetting_dossiers
from .injection import compact_grid, plot_recovery_map
from .ml import train_models
from .tess import parse_tic_id
from .pipeline import analyse_target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve and preprocess public TESS PDCSAP data.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyse = subparsers.add_parser("analyze", help="Retrieve and process one TIC.")
    analyse.add_argument("tic_id", help='TIC ID, for example "TIC 307210830".')
    analyse.add_argument("--output-root", default="outputs/targets", help="Directory for target outputs.")
    analyse.add_argument("--skip-tls", action="store_true", help="Skip optional TLS refinement.")
    vet = subparsers.add_parser("vet", help="Create Commit 3 vetting dossiers for an existing target run.")
    vet.add_argument("tic_id", help='TIC ID, for example "TIC 402026209".')
    vet.add_argument("--output-root", default="outputs/targets", help="Directory containing target outputs.")
    labels = subparsers.add_parser("label-manifest", help="Cache public labels separately from ExoSignal features.")
    labels.add_argument("--output", default="outputs/benchmark/label_manifest.csv")
    labels.add_argument("--per-class", type=int, default=100)
    labels.add_argument("--supplemental-negatives", type=int, help="Create this many non-overlapping reliable FP/FA TICs.")
    labels.add_argument("--exclude-manifest", help="CSV manifest whose TICs are excluded from a supplementary sample.")
    benchmark = subparsers.add_parser("benchmark", help="Build a labelled feature table from a manifest.")
    benchmark.add_argument("manifest", help="CSV made by label-manifest; labels are evaluation-only.")
    benchmark.add_argument("--output-directory", default="outputs/benchmark")
    benchmark.add_argument("--exclude-tics", default="", help="Comma-separated TICs deliberately omitted (for example, a retrieval timeout).")
    benchmark.add_argument("--maximum-targets", type=int, help="Process a deterministic leading batch; rerun with a later manifest slice to continue.")
    benchmark.add_argument("--target-offset", type=int, default=0, help="Zero-based manifest offset for a deterministic batch.")
    benchmark.add_argument("--workers", type=int, default=1, help="Independent TIC retrieval workers (use a small number, e.g. 4).")
    train = subparsers.add_parser("train", help="Train logistic and calibrated-RF models on an existing feature table.")
    train.add_argument("features", help="Benchmark feature CSV created by benchmark.")
    train.add_argument("--output-directory", default="outputs/models")
    discover = subparsers.add_parser("discover", help="Run a catalogue-blind MAST sector discovery sample.")
    discover.add_argument("--sector", required=True, type=int)
    discover.add_argument("--maximum-targets", required=True, type=int)
    discover.add_argument("--model", required=True, help="Saved ranking-model joblib.")
    discover.add_argument("--output-directory", default="outputs/discovery")
    discover.add_argument("--tmag-limit", type=float)
    discover.add_argument("--exclude-manifest", action="append", default=[], help="Label-manifest CSV whose TICs must be excluded (repeat for supplements).")
    reveal = subparsers.add_parser("reveal", help="Crossmatch an already frozen discovery ranking afterwards.")
    reveal.add_argument("ranking", help="pre_crossmatch_ranking.csv")
    reveal.add_argument("--output", default="outputs/discovery/catalogue_reveal.csv")
    inject = subparsers.add_parser("inject", help="Run a compact real-data injection/recovery grid for an existing run.")
    inject.add_argument("tic_id")
    inject.add_argument("--output-root", default="outputs/targets")
    inject.add_argument("--periods", default="1.5,3.0,7.0")
    inject.add_argument("--depths", default="0.001,0.005,0.01")
    inject.add_argument("--duration-hours", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "analyze":
            summary = analyse_target(
                args.tic_id, args.output_root, search_config=SearchConfig(tls_enabled=not args.skip_tls)
            )
        elif args.command == "vet":
            summary = run_vetting_dossiers(parse_tic_id(args.tic_id), args.output_root)
        elif args.command == "label-manifest":
            output = Path(args.output)
            catalogue = download_toi_catalogue(output.parent / "exofop_toi_catalogue.psv")
            if args.supplemental_negatives is not None:
                if not args.exclude_manifest:
                    raise ExoSignalError("--supplemental-negatives requires --exclude-manifest.")
                existing = set(pd.read_csv(args.exclude_manifest)["tic_id"].astype(int))
                table = supplemental_negative_manifest(catalogue, existing, args.supplemental_negatives)
            else:
                table = benchmark_manifest(catalogue, args.per_class)
            output.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(output, index=False)
            summary = {"label_manifest": str(output), "targets": len(table), "labels_are_not_model_features": True}
        elif args.command == "benchmark":
            excluded = {int(value) for value in args.exclude_tics.split(",") if value.strip()}
            output = Path(args.output_directory)
            output.mkdir(parents=True, exist_ok=True)
            # One writer owns a cumulative benchmark directory at a time.
            # This matters because resumed batches update shared CSV ledgers.
            with (output / ".benchmark.lock").open("w", encoding="utf-8") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                summary = build_benchmark(pd.read_csv(args.manifest), args.output_directory, exclude_tic_ids=excluded, maximum_targets=args.maximum_targets, target_offset=args.target_offset, workers=args.workers)
        elif args.command == "train":
            summary = train_models(pd.read_csv(args.features), args.output_directory)
        elif args.command == "discover":
            manifest_tics: set[int] = set()
            for manifest_path in args.exclude_manifest:
                manifest_tics.update(pd.read_csv(manifest_path)["tic_id"].astype(int))
            summary = run_discovery(args.sector, args.maximum_targets, args.model, args.output_directory, args.tmag_limit, additional_excluded_tic_ids=manifest_tics)
        elif args.command == "reveal":
            ranking = pd.read_csv(args.ranking)
            output = Path(args.output)
            catalogue = download_toi_catalogue(output.parent / "exofop_toi_catalogue.psv")
            revealed = reveal_catalogue_matches(ranking, catalogue)
            output.parent.mkdir(parents=True, exist_ok=True)
            revealed.to_csv(output, index=False)
            summary = {"catalogue_reveal": str(output), "signals": len(revealed), "no_match_in_queried_exofop_toi_catalogue": int(revealed.get("catalogue_reveal", pd.Series(dtype=str)).eq("NO_MATCH_IN_QUERIED_EXOFOP_TOI_CATALOGUE").sum())}
        else:
            tic = parse_tic_id(args.tic_id)
            data = pd.read_csv(Path(args.output_root) / str(tic) / "processed_lightcurve_initial.csv")
            # Sensitivity is deliberately compact: it runs the normal BLS
            # path, but a capped period grid keeps a 3x3 real-data study
            # practical on a laptop. It is not used for candidate reporting.
            injection_search = SearchConfig(tls_enabled=False, max_candidates=3, max_bls_period_samples=8_000)
            results = compact_grid(data, [float(value) for value in args.periods.split(",")], [float(value) for value in args.depths.split(",")], args.duration_hours, float(data["time"].median()), search_config=injection_search)
            output = Path(args.output_root) / str(tic) / "injection_recovery.csv"
            results.to_csv(output, index=False)
            plot = output.with_suffix(".png")
            plot_recovery_map(results, plot)
            summary = {
                "injection_recovery": str(output), "sensitivity_plot": str(plot), "trials": len(results),
                "exact_period_recoveries": int(results["recovery"].eq("exact_period").sum()),
                "harmonic_recoveries": int(results["recovery"].eq("simple_harmonic").sum()),
                "non_recoveries": int(results["recovery"].isin(["not_recovered", "other_period"]).sum()),
            }
    except ExoSignalError as error:
        print(f"ExoSignal error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
