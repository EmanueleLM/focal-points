import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from bargaining_table import plot_pretty_boxplot


FILE_RE = re.compile(
    r"(?P<strategy>p[12]_llm)_(?P<color>blue|yellow)_"
    r"(?P<model>gpt-oss-(?:20b|120b))-(?P<reasoning>low|medium|high)_"
    r"(?P<dataset>bargaining_table(?:_realdata)?)_responses_"
    r"(?P<variant>[a-z-]+)_(?P=color)\.json$"
)

DEFAULT_VARIANTS = [
    "vanilla",
    "saliency",
    "greedy",
    "cooperative",
    "all-features",
]

DEFAULT_REASONING = ["low", "medium", "high"]
DEFAULT_MODELS = ["gpt-oss-120b", "gpt-oss-20b"]
DEFAULT_STRATEGIES = ["p1_llm", "p2_llm"]

STRATEGY_LABELS = {
    "p1_llm": ("Orange Human", "Blue LLM"),
    "p2_llm": ("Orange LLM", "Blue Human"),
}


@dataclass(frozen=True)
class ResultMeta:
    strategy: str
    color: str
    model: str
    reasoning: str
    dataset: str
    variant: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot averaged bargaining-table results for gpt-oss models."
    )
    parser.add_argument(
        "--results-dir",
        default="./results/bargaining_table",
        help="Base folder containing bargaining_table result JSONs.",
    )
    parser.add_argument(
        "--plots-dir",
        default="./plots/bargaining_table/averages",
        help="Output folder for plots.",
    )
    parser.add_argument(
        "--dataset",
        default="bargaining_table_realdata",
        help="Dataset prefix to filter on (e.g. bargaining_table_realdata).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Prompt variants to include (default: all available).",
    )
    parser.add_argument(
        "--reasoning-levels",
        nargs="+",
        default=DEFAULT_REASONING,
        help="Reasoning levels to include (default: low medium high).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names to include (default: gpt-oss-120b gpt-oss-20b).",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=DEFAULT_STRATEGIES,
        help="Strategies to include (default: p1_llm p2_llm).",
    )
    return parser.parse_args()


def scan_results(results_dir: Path, dataset_filter: Optional[str]) -> List[ResultMeta]:
    entries: List[ResultMeta] = []
    for path in results_dir.rglob("*.json"):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        info = match.groupdict()
        if dataset_filter and info["dataset"] != dataset_filter:
            continue
        entries.append(
            ResultMeta(
                strategy=info["strategy"],
                color=info["color"],
                model=info["model"],
                reasoning=info["reasoning"],
                dataset=info["dataset"],
                variant=info["variant"],
                path=path,
            )
        )
    return entries


def build_index(entries: Iterable[ResultMeta]) -> Dict[Tuple[str, str, str, str, str], List[Path]]:
    index: Dict[Tuple[str, str, str, str, str], List[Path]] = {}
    for entry in entries:
        key = (entry.strategy, entry.model, entry.reasoning, entry.variant, entry.dataset)
        index.setdefault(key, []).append(entry.path)
    return index


def resolve_variants(entries: Iterable[ResultMeta], requested: Optional[List[str]]) -> List[str]:
    available = sorted({entry.variant for entry in entries})
    if requested:
        return [v for v in requested if v in available]
    ordered = [v for v in DEFAULT_VARIANTS if v in available]
    ordered.extend([v for v in available if v not in ordered])
    return ordered


def pick_single_path(paths: Optional[List[Path]], key: Tuple[str, str, str, str, str]) -> Optional[Path]:
    if not paths:
        return None
    if len(paths) > 1:
        names = ", ".join(p.name for p in paths)
        print(f"Warning: multiple results for {key}: {names}. Using {paths[0].name}.")
    return paths[0]


def load_total_payoffs(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    keys_p1 = sorted(data["overall_payoff_p1"].keys(), key=int)
    keys_p2 = sorted(data["overall_payoff_p2"].keys(), key=int)
    games_p1 = np.array([data["overall_payoff_p1"][k] for k in keys_p1])
    games_p2 = np.array([data["overall_payoff_p2"][k] for k in keys_p2])
    return games_p1.sum(axis=0), games_p2.sum(axis=0)


def average_arrays(arrays: List[np.ndarray], label: str) -> np.ndarray:
    if not arrays:
        raise ValueError(f"No arrays provided for {label}.")
    lengths = [len(a) for a in arrays]
    min_len = min(lengths)
    if len(set(lengths)) > 1:
        print(f"Warning: {label} arrays have lengths {lengths}; trimming to {min_len}.")
    stacked = np.vstack([a[:min_len] for a in arrays])
    return stacked.mean(axis=0)


def average_results(paths: List[Path], label: str) -> Tuple[np.ndarray, np.ndarray]:
    totals_p1: List[np.ndarray] = []
    totals_p2: List[np.ndarray] = []
    for path in paths:
        p1, p2 = load_total_payoffs(path)
        totals_p1.append(p1)
        totals_p2.append(p2)
    return (
        average_arrays(totals_p1, f"{label} p1"),
        average_arrays(totals_p2, f"{label} p2"),
    )


def build_labels(strategy: str, llm_detail: Optional[str]) -> Tuple[str, str]:
    orange_label, blue_label = STRATEGY_LABELS.get(strategy, ("Player 2", "Player 1"))
    if llm_detail:
        if strategy == "p1_llm":
            blue_label = f"{blue_label} ({llm_detail})"
        elif strategy == "p2_llm":
            orange_label = f"{orange_label} ({llm_detail})"
    return blue_label, orange_label


def plot_and_save(
    data_p1: np.ndarray,
    data_p2: np.ndarray,
    strategy: str,
    save_path: Path,
    llm_detail: Optional[str],
) -> None:
    label1, label2 = build_labels(strategy, llm_detail)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_pretty_boxplot(
        data_p1,
        data_p2,
        label1=label1,
        label2=label2,
        annotate_stats=True,
        save_path=str(save_path),
    )


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    entries = scan_results(results_dir, args.dataset)
    if not entries:
        print("No matching results found.")
        return

    variants = resolve_variants(entries, args.variants)
    strategies = [s for s in args.strategies if s in {e.strategy for e in entries}]
    if not strategies:
        print("No matching strategies found.")
        return
    available_models = sorted({e.model for e in entries})
    models = [m for m in args.models if m in available_models]
    if not models:
        print("No matching models found.")
        return
    models_tag = "+".join(m.replace("gpt-oss-", "") for m in models)

    index = build_index(entries)

    # Average each model across reasoning levels for each variant.
    for strategy in strategies:
        for variant in variants:
            for model in models:
                paths: List[Path] = []
                for reasoning in args.reasoning_levels:
                    key = (strategy, model, reasoning, variant, args.dataset)
                    path = pick_single_path(index.get(key), key)
                    if path:
                        paths.append(path)
                if not paths:
                    print(
                        f"Missing {model} results for {strategy} {variant}. Skipping."
                    )
                    continue
                avg_p1, avg_p2 = average_results(
                    paths, f"{strategy} {model} {variant}"
                )
                save_path = (
                    plots_dir
                    / "avg-model-across-reasoning"
                    / strategy
                    / f"{args.dataset}_{variant}_avg-{model}-across-reasoning.png"
                )
                plot_and_save(
                    avg_p1,
                    avg_p2,
                    strategy,
                    save_path,
                    llm_detail=f"{model} avg reasoning",
                )

    # Average across models for each reasoning level and variant.
    for strategy in strategies:
        for variant in variants:
            for reasoning in args.reasoning_levels:
                paths = []
                for model in models:
                    key = (strategy, model, reasoning, variant, args.dataset)
                    path = pick_single_path(index.get(key), key)
                    if path:
                        paths.append(path)
                if not paths:
                    print(
                        f"Missing model results for {strategy} {variant} {reasoning}. Skipping."
                    )
                    continue
                avg_p1, avg_p2 = average_results(
                    paths, f"{strategy} {variant} {reasoning}"
                )
                save_path = (
                    plots_dir
                    / "avg-models-by-reasoning"
                    / strategy
                    / f"{args.dataset}_{variant}_avg-{models_tag}-{reasoning}.png"
                )
                plot_and_save(
                    avg_p1,
                    avg_p2,
                    strategy,
                    save_path,
                    llm_detail=f"avg {models_tag} {reasoning}",
                )


if __name__ == "__main__":
    main()
