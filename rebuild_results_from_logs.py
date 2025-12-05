import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

from src.utils import plot_block_frequencies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild plots/results from existing response logs."
    )
    parser.add_argument(
        "--logs-root",
        default="./logs",
        help="Root directory containing log files (default: ./logs).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help=(
            "Optional list of model names to include "
            "(e.g., 'openai/gpt-oss-120b_low'). If omitted, process all models."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the jobs that would run without executing them.",
    )
    return parser.parse_args()


def discover_jobs(
    logs_root: Path, allowed_models: List[str] | None
) -> Iterable[Tuple[str, Path, str, str]]:
    """
    Yield (model_name, log_file, dataset, problem_tag) tuples for every log file
    that matches the expected naming pattern: <dataset>_responses_<problem>.jsonl.
    """
    pattern = re.compile(r"(.+)_responses_(.+)\.jsonl$")

    for log_file in logs_root.rglob("*_responses_*.jsonl"):
        match = pattern.match(log_file.name)
        if not match:
            continue

        dataset, problem_tag = match.group(1), match.group(2)
        model_name = log_file.parent.relative_to(logs_root).as_posix()

        if allowed_models and model_name not in allowed_models:
            continue

        yield model_name, log_file, dataset, problem_tag


def main() -> int:
    args = parse_args()
    logs_root = Path(args.logs_root)

    if not logs_root.exists():
        print(f"Logs root does not exist: {logs_root}")
        return 1

    allowed_models = args.models if args.models else None
    jobs = sorted(
        discover_jobs(logs_root, allowed_models),
        key=lambda j: (j[0], j[2], j[3]),
    )

    if not jobs:
        print("No matching log files found.")
        return 1

    for model_name, log_file, dataset, problem_tag in jobs:
        print(
            f"[{model_name}] dataset={dataset}, problem_tag={problem_tag} "
            f"(log: {log_file})"
        )

        if args.dry_run:
            continue

        with open(log_file, "r") as f:
            data = json.load(f)

        try:
            os.makedirs(f"./images/{model_name}/{dataset}/{problem_tag}/", exist_ok=True)
            os.makedirs(f"./results/{model_name}/", exist_ok=True)
            plot_block_frequencies(data, dataset, model_name, problem_tag)
        except Exception as exc:  # keep going on failure
            print(f"Failed on {log_file}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
