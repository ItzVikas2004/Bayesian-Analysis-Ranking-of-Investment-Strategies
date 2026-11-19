"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging

from .pipeline import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Bayesian investment strategy research")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ["run", "report"]:
        p = sub.add_parser(command)
        p.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.command in {"run", "report"}:
        run_experiment(args.config)


if __name__ == "__main__":
    main()
