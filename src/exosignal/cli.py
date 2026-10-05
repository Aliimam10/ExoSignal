"""Small command-line entry point for Commit 1."""

from __future__ import annotations

import argparse
import json
import sys

from .errors import ExoSignalError
from .config import SearchConfig
from .pipeline import analyse_target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve and preprocess public TESS PDCSAP data.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyse = subparsers.add_parser("analyze", help="Retrieve and process one TIC.")
    analyse.add_argument("tic_id", help='TIC ID, for example "TIC 307210830".')
    analyse.add_argument("--output-root", default="outputs/targets", help="Directory for target outputs.")
    analyse.add_argument("--skip-tls", action="store_true", help="Skip optional TLS refinement.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        summary = analyse_target(
            args.tic_id, args.output_root, search_config=SearchConfig(tls_enabled=not args.skip_tls)
        )
    except ExoSignalError as error:
        print(f"ExoSignal error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
