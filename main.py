import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from src.llm import LLM
from src.prompt import Level0
from src.utils import iterate_data


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model_name",
                        default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model id string.")
    parser.add_argument("-d", "--dataset", dest="dataset",
                        default="schelling", help="Dataset name (without .jsonl).")
    parser.add_argument("-p", "--problem-tag", dest="problem_tag",
                        default="problem", help="Key of the problem in the json data.")
    parser.add_argument("-t", "--trials", dest="trials", type=int, default=1,
                        help="How many times to call the model per prompt.")
    parser.add_argument("-s", "--return-sequences", dest="sequences", type=int,
                        default=30, help="Responses per single prompt.")
    parser.add_argument("-q", "--quantization", dest="quantization",
                        default=None, help="None, 8bit or 4bit.")
    return parser.parse_args()


def prepare_directories(model_name: str, base_data_dir: str = "./data/") -> Tuple[Path, Path]:
    dataset_dir = Path(base_data_dir)
    logs_dir = Path("./logs") / model_name
    for d in (dataset_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    return dataset_dir, logs_dir


def save_jsonl(path: Path, data: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def build_flat_prompt_list(problems: Dict[int, List[str]]) -> Tuple[List[str], List[Tuple[int, str]]]:
    prompts, keys = [], []
    for idx, variants in problems.items():
        for v in variants:
            prompts.append(f"{Level0.prefix}{v}{Level0.suffix}")
            keys.append((idx, v))
    return prompts, keys


def generate_batch_responses(model: LLM, prompts: List[str], trials: int, seq_per_prompt: int) -> List[List[str]]:
    all_outputs: List[List[str]] = []
    for i, prompt in enumerate(prompts, start=1):
        combined: List[str] = []
        for t in range(trials):
            print(f"[{model.model_id}] Q {i}/{len(prompts)}  "
                  f"Trial {t + 1}/{trials}  "
                  f"({seq_per_prompt} seqs)")
            texts = model.generate_batch([prompt])[0]
            combined.extend(texts)
            print("\n".join(texts))
        all_outputs.append(combined)
    return all_outputs


def nest_outputs(keys: List[Tuple[int, str]], outputs: List[List[str]]) -> Dict[int, Dict[str, List[str]]]:
    resp: Dict[int, Dict[str, List[str]]] = {}
    for (idx, prompt), outs in zip(keys, outputs):
        resp.setdefault(idx, {})[prompt] = outs
    return resp


def build_log_json(idx_to_responses: Dict[int, Dict[str, List[str]]], norm_factors: Dict[int, List[int]]) -> List[dict]:
    logs: List[dict] = []
    for idx, variants in idx_to_responses.items():
        for (var_idx, prompt), norm in zip(enumerate(variants), norm_factors[idx]):
            logs.append({
                "idx": idx,
                "variation-idx": str(var_idx),
                "prompt": prompt,
                "responses": variants[prompt],
                "normalization_factor": norm,
            })
    return logs


def run_job(args: argparse.Namespace) -> None:
    start = time.time()

    # filesystem prep
    dataset_dir, logs_dir = prepare_directories(args.model_name)

    # load dataset
    ds_path = dataset_dir / f"{args.dataset}.jsonl"
    with open(ds_path) as f:
        raw_data = json.load(f)
    problems, norm_factors = iterate_data(raw_data, args.problem_tag)

    # load model
    model = LLM(
        model_id=args.model_name,
        num_return_sequences=args.sequences,
        quantization=args.quantization,
    )

    # build the prompt list
    prompts, keys = build_flat_prompt_list(problems)

    # generate
    outputs = generate_batch_responses(model, prompts, trials=args.trials, seq_per_prompt=args.sequences)

    # reshape & save
    nested = nest_outputs(keys, outputs)
    jsonl_logs = build_log_json(nested, norm_factors)
    out_file = logs_dir / f"{args.dataset}_responses_{args.problem_tag}.jsonl"
    save_jsonl(out_file, jsonl_logs)
    print(f"[OK] Responses written to {out_file}")

    # print timing
    elapsed = int(time.time() - start)
    print(f"[{args.model_name}] elapsed "
          f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}")

    # cleanup – free VRAM and delete weights on disk
    model.clear_cache()


if __name__ == "__main__":
    cli_args = parse_arguments()

    try:
        run_job(cli_args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
