import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
from src.llm import LLM, load_model
from src.prompt import Level0
from src.utils import iterate_data, load_bargaining_table_prompts, plot_block_frequencies


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        dest="model_names",
        nargs="+",
        default=["meta-llama/Llama-3.2-1B-Instruct"],
        help="HuggingFace model id string. Accepts multiple values.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="datasets",
        nargs="+",
        default=["schelling"],
        help="Dataset name(s) (without .jsonl).",
    )
    parser.add_argument(
        "-p",
        "--problem-tag",
        dest="problem_tags",
        nargs="+",
        default=["problem"],
        help="Key(s) of the problem in the json data.",
    )
    parser.add_argument(
        "-t",
        "--trials",
        dest="trials",
        type=int,
        default=1,
        help="How many times to call the model per prompt.",
    )
    parser.add_argument(
        "-s",
        "--return-sequences",
        dest="sequences",
        type=int,
        default=30,
        help="Responses per single prompt.",
    )
    parser.add_argument(
        "-x",
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of new tokens to generate per response.",
    )
    parser.add_argument(
        "-q",
        "--quantization",
        dest="quantization",
        default=None,
        help="None, 8bit or 4bit.",
    )
    parser.add_argument(
        "-g",
        "--plot-graphs",
        dest="plot_graph",
        type=lambda s: s.lower() == "true",
        default=True,
        help="Whether to plot or not the barplots for each model and problem.",
    )
    parser.add_argument(
        "-r",
        "--reasoning",
        dest="reasoning",
        default=None,
        help='Reasoning effort for API models (e.g., "low", "medium").',
    )
    parser.add_argument(
        "--bargaining-player",
        dest="bargaining_player",
        default="blue",
        type=lambda s: s.lower(),
        choices=["blue", "yellow"],
        help="Player role for bargaining table datasets.",
    )
    return parser.parse_args()


def ensure_list(values: str | List[str]) -> List[str]:
    return values if isinstance(values, list) else [values]


def prepare_directories(
    model_name: str, base_data_dir: str = "./data/"
) -> Tuple[Path, Path]:
    dataset_dir = Path(base_data_dir)
    logs_dir = Path("./logs") / model_name
    for d in (dataset_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    return dataset_dir, logs_dir


def save_jsonl(path: Path, data: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def build_prompt_plan(
    problems: Dict[int, List[str]], norm_factors: Dict[int, List[int]]
) -> List[dict]:
    prompt_plan: List[dict] = []
    for idx, variants in problems.items():
        for var_idx, variant in enumerate(variants):
            prompt_plan.append(
                {
                    "idx": idx,
                    "variation_idx": str(var_idx),
                    "prompt": f"{Level0.prefix}{variant}{Level0.suffix}",
                    "normalization_factor": norm_factors[idx][var_idx],
                }
            )
    return prompt_plan


def assemble_log_entries(
    prompt_plan: List[dict], responses_by_prompt: Dict[Tuple[int, str], List[str]]
) -> List[dict]:
    logs: List[dict] = []
    for entry in prompt_plan:
        key = (entry["idx"], entry["prompt"])
        logs.append(
            {
                "idx": entry["idx"],
                "variation-idx": entry["variation_idx"],
                "prompt": entry["prompt"],
                "responses": responses_by_prompt.get(key, []),
                "normalization_factor": entry["normalization_factor"],
            }
        )
    return logs


def load_existing_logs(path: Path) -> Dict[Tuple[int, str], List[str]]:
    if not path.exists():
        return {}

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"[WARNING] Could not parse existing log at {path}; ignoring it.")
        return {}

    responses: Dict[Tuple[int, str], List[str]] = {}
    for block in data:
        idx = block.get("idx")
        prompt = block.get("prompt")
        if idx is None or prompt is None:
            continue
        stored_responses = block.get("responses") or []
        if isinstance(stored_responses, list):
            filtered = [r for r in stored_responses if isinstance(r, str)]
            responses[(idx, prompt)] = filtered
    return responses


def build_flat_prompt_list(
    problems: Dict[int, List[str]],
) -> Tuple[List[str], List[Tuple[int, str]]]:
    prompts, keys = [], []
    for idx, variants in problems.items():
        for v in variants:
            prompts.append(f"{Level0.prefix}{v}{Level0.suffix}")
            keys.append((idx, v))
    return prompts, keys


def generate_batch_responses(
    model: LLM,
    prompt_plan: List[dict],
    trials: int,
    seq_per_prompt: int,
    log_path: Path,
    existing_responses: Dict[Tuple[int, str], List[str]],
) -> Tuple[Dict[Tuple[int, str], List[str]], bool]:
    target_per_prompt = trials * seq_per_prompt
    responses_by_prompt: Dict[Tuple[int, str], List[str]] = {
        key: list(val) for key, val in existing_responses.items()
    }

    for entry in prompt_plan:
        responses_by_prompt.setdefault((entry["idx"], entry["prompt"]), [])

    new_data_generated = False
    total_prompts = len(prompt_plan)

    for i, entry in enumerate(prompt_plan, start=1):
        key = (entry["idx"], entry["prompt"])
        prompt_responses = responses_by_prompt[key]

        if len(prompt_responses) >= target_per_prompt:
            print(
                f"[SKIP] idx={entry['idx']} var={entry['variation_idx']} "
                f"already has {len(prompt_responses)}/{target_per_prompt} responses."
            )
            continue

        if prompt_responses:
            print(
                f"[RESUME] idx={entry['idx']} var={entry['variation_idx']} "
                f"continuing from {len(prompt_responses)}/{target_per_prompt}."
            )

        while len(prompt_responses) < target_per_prompt:
            remaining = target_per_prompt - len(prompt_responses)
            batch_size = min(seq_per_prompt, remaining)
            print(
                f"[{model.model_id}] Q {i}/{total_prompts} "
                f"idx={entry['idx']} var={entry['variation_idx']} "
                f"({len(prompt_responses)}/{target_per_prompt} done, "
                f"+{batch_size} requested)"
            )
            print("[PROMPT]")
            print(entry["prompt"])
            if model.is_api_model:
                print("[RESPONSES]")
                for _ in range(batch_size):
                    text = model.generate(entry["prompt"])
                    prompt_responses.append(text)
                    new_data_generated = True
                    print(text)
                    save_jsonl(
                        log_path,
                        assemble_log_entries(prompt_plan, responses_by_prompt),
                    )
                continue

            texts = model.generate_batch(
                [entry["prompt"]], num_return_sequences=batch_size
            )[0]
            prompt_responses.extend(texts[:batch_size])
            new_data_generated = True
            print("[RESPONSES]")
            print("\n".join(texts[:batch_size]))

            save_jsonl(log_path, assemble_log_entries(prompt_plan, responses_by_prompt))

    return responses_by_prompt, new_data_generated


def nest_outputs(
    keys: List[Tuple[int, str]], outputs: List[List[str]]
) -> Dict[int, Dict[str, List[str]]]:
    resp: Dict[int, Dict[str, List[str]]] = {}
    for (idx, prompt), outs in zip(keys, outputs):
        resp.setdefault(idx, {})[prompt] = outs
    return resp


def build_log_json(
    idx_to_responses: Dict[int, Dict[str, List[str]]],
    norm_factors: Dict[int, List[int]],
) -> List[dict]:
    logs: List[dict] = []
    for idx, variants in idx_to_responses.items():
        for (var_idx, prompt), norm in zip(enumerate(variants), norm_factors[idx]):
            logs.append(
                {
                    "idx": idx,
                    "variation-idx": str(var_idx),
                    "prompt": prompt,
                    "responses": variants[prompt],
                    "normalization_factor": norm,
                }
            )
    return logs


def run_single_job(
    model_name: str,
    dataset: str,
    problem_tag: str,
    args: argparse.Namespace,
    cached_model: LLM | None,
) -> LLM | None:
    start = time.time()

    # filesystem prep
    dataset_dir, logs_dir = prepare_directories(model_name)
    log_suffix = problem_tag
    if dataset in ("bargaining_table", "bargaining_table_realdata"):
        log_suffix = f"{problem_tag}_{args.bargaining_player}"
    log_file = logs_dir / f"{dataset}_responses_{log_suffix}.jsonl"

    # load dataset
    if dataset in ("bargaining_table", "bargaining_table_realdata"):
        ds_path = dataset_dir / "bargaining_table_llms" / f"{dataset}.json"
        problems, norm_factors = load_bargaining_table_prompts(
            ds_path, args.bargaining_player, problem_tag
        )
    else:
        ds_path = dataset_dir / f"{dataset}.jsonl"
        with open(ds_path) as f:
            raw_data = json.load(f)
        problems, norm_factors = iterate_data(raw_data, problem_tag)

    # build prompt plan and check existing progress before loading the model
    prompt_plan = build_prompt_plan(problems, norm_factors)
    target_per_prompt = args.trials * args.sequences
    existing_responses = load_existing_logs(log_file)
    total_prompts = len(prompt_plan)
    completed_prompts = sum(
        1
        for entry in prompt_plan
        if len(existing_responses.get((entry["idx"], entry["prompt"]), []))
        >= target_per_prompt
    )
    if existing_responses:
        print(
            f"[INFO] Found stored responses for {completed_prompts}/{total_prompts} "
            f"prompts in {log_file}."
        )

    all_complete = completed_prompts == total_prompts and total_prompts > 0
    responses_by_prompt: Dict[Tuple[int, str], List[str]] = dict(existing_responses)
    model = cached_model
    new_data_generated = False

    # load model if additional work is required
    if not all_complete:
        if total_prompts == 0:
            print("[WARNING] No prompts generated from dataset; exiting early.")
            return model

        # load model once per model_name and reuse for subsequent jobs
        if model is None:
            reasoning_arg = args.reasoning
            if reasoning_arg and reasoning_arg.lower() == "none":
                reasoning_arg = None

            model = load_model(
                model_id=model_name,
                num_return_sequences=args.sequences,
                max_new_tokens=args.max_new_tokens,
                quantization=args.quantization,
                reasoning=reasoning_arg,
            )

        # generate
        responses_by_prompt, new_data_generated = generate_batch_responses(
            model,
            prompt_plan,
            trials=args.trials,
            seq_per_prompt=args.sequences,
            log_path=log_file,
            existing_responses=existing_responses,
        )
    else:
        print("[INFO] All prompts already have the required number of responses.")

    # reshape & save
    jsonl_logs = assemble_log_entries(prompt_plan, responses_by_prompt)
    save_jsonl(log_file, jsonl_logs)
    print(f"[OK] Responses written to {log_file}")

    # print timing
    elapsed = int(time.time() - start)
    print(
        f"[{model_name}] elapsed "
        f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"
    )

    # Plot graphs and create result folders
    output_dataset = dataset
    if dataset in ("bargaining_table", "bargaining_table_realdata"):
        output_dataset = f"{dataset}_{args.bargaining_player}"
    results_path = Path(
        f"./results/{model_name}/{output_dataset}_{problem_tag}.jsonl"
    )
    if args.plot_graph:
        if new_data_generated or not results_path.exists():
            compute_metrics(model_name, output_dataset, jsonl_logs, problem_tag)
        else:
            print(
                f"[INFO] Results already exist at {results_path}; skipping recomputation."
            )

    return model


def run_job(args: argparse.Namespace) -> None:
    for model in ensure_list(args.model_names):
        cached_model: LLM | None = None
        for dataset in ensure_list(args.datasets):
            problem_tags = ensure_list(args.problem_tags)
            if (
                dataset in ("bargaining_table", "bargaining_table_realdata")
                and problem_tags == ["problem"]
            ):
                ds_path = Path("./data") / "bargaining_table_llms" / f"{dataset}.json"
                with open(ds_path, "r") as f:
                    data = json.load(f)
                variants = data.get("variants", {})
                if not isinstance(variants, dict) or not variants:
                    raise ValueError(
                        f"Dataset at {ds_path} must include a non-empty 'variants' mapping."
                    )
                problem_tags = list(variants.keys())
            for problem_tag in problem_tags:
                print(
                    f"[INFO] Running model={model} dataset={dataset} "
                    f"problem_tag={problem_tag}"
                )
                cached_model = run_single_job(
                    model, dataset, problem_tag, args, cached_model
                )
        if cached_model and hasattr(cached_model, "clear_cache"):
            cached_model.clear_cache()


def compute_metrics(
    model_name: str, dataset: str, data: List[dict], problem_tag: str
) -> None:
    """Plot graphs and compute the results

    Args:
        model_name (str): the LLM name
        dataset (str): the dataset name
        problem_tag (str): pick, guess, or coordinate
    """
    os.makedirs(f"./images/{model_name}/{dataset}/{problem_tag}/", exist_ok=True)
    os.makedirs(f"./results/{model_name}/", exist_ok=True)

    plot_block_frequencies(data, dataset, model_name, problem_tag)


if __name__ == "__main__":
    # Parse CLI args and run the job
    cli_args = parse_arguments()

    run_job(cli_args)
