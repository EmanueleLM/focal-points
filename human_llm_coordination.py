import json
import argparse
from pathlib import Path 
import matplotlib.pyplot as plt
import numpy as np
from difflib import SequenceMatcher
from typing import Optional

def load_human_data(file_path, task):
    """Load human data from a JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    results = {}
    # Retrieve the frequency of each unique response per problem
    for entry in data:
        results[entry["id"]] = {c.lower(): freq for (c, freq) in zip(entry["choices"], entry[task])}

    return results


def load_llm_data(file_path, human_data):
    """Load LLM data from a JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Retrieve the frequency of each unique response per problem
    results = {}
    for idx in human_data.keys():
        # Collect corresponding LLM entries (there can be multiple)
        llm_entries = [entry for entry in data if entry["idx"] == idx]
        
        # Merge all entries and cumulative frequencies
        results[idx] = {}
        for llm_entry in llm_entries:
            for key, freq in llm_entry["responses"].items():
                try:
                    filtered_key = key.split("<answer>")[1].split("</answer>")[0].strip().lower()
                    if filtered_key not in results[idx]:
                        results[idx][filtered_key] = 0
                    results[idx][filtered_key] += freq
                except:
                    pass 
    return results


def normalize_key(k):
    """Lowercase, strip and remove trailing digits (common OCR artifacts)."""
    import re
    k = str(k).strip().lower()
    k = re.sub(r'\d+$', '', k)  # remove trailing digits
    return k.strip()


def fuzzy_map_keys(source_keys, target_keys, threshold=0.82):
    """
    Map each source_key (usually from llm) to the best matching key in target_keys (human).
    Returns dict source_key -> matched_target_key or None if no match above threshold.
    Ensures each target key is only mapped once (greedy by best ratio).
    """
    mapping = {}
    used_targets = set()
    for s in source_keys:
        best = None
        best_ratio = 0.0
        for t in target_keys:
            if t in used_targets:
                continue
            ratio = SequenceMatcher(None, s, t).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best = t
        if best is not None and best_ratio >= threshold:
            mapping[s] = best
            used_targets.add(best)
        else:
            mapping[s] = None
    return mapping


def normalized_distribution(d):
    """L1-normalize dict values to sum to 1. Returns empty dict for empty input."""
    if not d:
        return {}
    total = float(sum(d.values()))
    if total == 0:
        return {k: 0.0 for k in d}
    return {k: v/total for k, v in d.items()}


def js_divergence_for_task(human_d, llm_d, use_fuzzy=True, fuzzy_threshold=0.82):
    """
    Compute a symmetric KL-based divergence (Jensen–Shannon) between
    the human and LLM response distributions for a single task.

    - human_d, llm_d: dicts mapping item -> count (may be empty)
    - use_fuzzy: if True, attempt to align llm keys to human keys using fuzzy matching
    - fuzzy_threshold: similarity threshold for accepting a fuzzy match (0..1)

    Returns: (js_divergence, diagnostics_dict)
      js_divergence is the Jensen–Shannon divergence between the L1-normalized
      distributions (in bits, using log base 2), bounded in [0, 1]. Lower is
      better (more similar / larger "overlap").
      diagnostics_dict contains normalized dicts and mapping used.
    """
    # normalize keys
    human = {normalize_key(k): v for k, v in (human_d or {}).items()}
    llm = {normalize_key(k): v for k, v in (llm_d or {}).items()}

    # empty-case handling: if either side is empty, return maximal divergence (1.0 in bits)
    if not human or not llm:
        return 1.0, {'human': human, 'llm': llm, 'mapped_llm': None}

    # optionally fuzzy-match llm keys to human keys
    mapped_llm = {}
    if use_fuzzy:
        mapping = fuzzy_map_keys(list(llm.keys()), list(human.keys()), threshold=fuzzy_threshold)
        for s_key, mapped in mapping.items():
            if mapped is None:
                # keep original key (distinct)
                mapped_llm[s_key] = mapped_llm.get(s_key, 0) + llm[s_key]
            else:
                mapped_llm[mapped] = mapped_llm.get(mapped, 0) + llm[s_key]
    else:
        mapped_llm = dict(llm)

    # union keys across human and mapped llm
    all_keys = sorted(set(list(human.keys()) + list(mapped_llm.keys())))
    h_norm = normalized_distribution({k: human.get(k, 0.0) for k in all_keys})
    l_norm = normalized_distribution({k: mapped_llm.get(k, 0.0) for k in all_keys})

    # Jensen–Shannon divergence (symmetric KL) in base 2
    # JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q)
    def _kl_divergence(p, q):
        # KL(p||q) with 0 * log(0/·) treated as 0; logs in base 2
        kl = 0.0
        for k in all_keys:
            pk = p.get(k, 0.0)
            qk = q.get(k, 0.0)
            if pk > 0.0:
                # qk can be zero only if pk==0 and l_norm[k]==0 as well, but guard anyway
                if qk <= 0.0:
                    # In theory this cannot happen for JSD since q is a mixture, but keep safe
                    return float('inf')
                kl += pk * (np.log2(pk) - np.log2(qk))
        return float(kl)

    m = {k: 0.5 * (h_norm.get(k, 0.0) + l_norm.get(k, 0.0)) for k in all_keys}
    kl_h_m = _kl_divergence(h_norm, m)
    kl_l_m = _kl_divergence(l_norm, m)
    js_divergence = 0.5 * (kl_h_m + kl_l_m)

    diagnostics = {
        'human': human,
        'llm': llm,
        'mapped_llm': mapped_llm,
        'all_keys': all_keys,
        'h_norm': h_norm,
        'l_norm': l_norm,
        'metric': 'jensen-shannon-divergence-bits'
    }
    return js_divergence, diagnostics


def overlap_for_task(human_d, llm_d, use_fuzzy=True, fuzzy_threshold=0.82):
    """Deprecated alias for js_divergence_for_task.
    Computes and returns the Jensen–Shannon divergence (lower is better).
    """
    return js_divergence_for_task(human_d, llm_d, use_fuzzy=use_fuzzy, fuzzy_threshold=fuzzy_threshold)


def _align_counts_for_problem(human_d, llm_d, use_fuzzy=True, fuzzy_threshold=0.82):
    """Align keys (with optional fuzzy mapping) and return integer count arrays.
    Returns: (all_keys, human_counts:int ndarray, llm_counts:int ndarray)
    """
    human = {normalize_key(k): int(v) for k, v in (human_d or {}).items()}
    llm = {normalize_key(k): int(v) for k, v in (llm_d or {}).items()}

    # map llm keys onto human keys if requested
    if use_fuzzy and human and llm:
        mapped_llm = {}
        mapping = fuzzy_map_keys(list(llm.keys()), list(human.keys()), threshold=fuzzy_threshold)
        for s_key, mapped in mapping.items():
            target = mapped if mapped is not None else s_key
            mapped_llm[target] = mapped_llm.get(target, 0) + llm[s_key]
    else:
        mapped_llm = dict(llm)

    all_keys = sorted(set(human.keys()) | set(mapped_llm.keys()))
    h_counts = np.array([int(human.get(k, 0)) for k in all_keys], dtype=int)
    l_counts = np.array([int(mapped_llm.get(k, 0)) for k in all_keys], dtype=int)
    return all_keys, h_counts, l_counts


def _js_from_counts(h_counts: np.ndarray, l_counts: np.ndarray) -> float:
    """Compute JSD (bits) from integer count vectors of same length.
    Returns 1.0 if either vector sums to zero.
    """
    h_sum = int(np.sum(h_counts))
    l_sum = int(np.sum(l_counts))
    if h_sum <= 0 or l_sum <= 0:
        return 1.0
    p = h_counts.astype(float) / h_sum
    q = l_counts.astype(float) / l_sum
    m = 0.5 * (p + q)
    # KL with mask to avoid 0 * log(0/·)
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * (np.log2(a[mask]) - np.log2(b[mask]))))
    return 0.5 * (_kl(p, m) + _kl(q, m))


def compute_overlaps_all_tasks(human_data, llm_data, use_fuzzy=True, fuzzy_threshold=0.82,
                               n_resamples: int = 1000, random_state: Optional[int] = None):
    """
    Compute per-problem Jensen–Shannon divergences and aggregate statistics.
    Also estimate a Monte Carlo p-value for the mean divergence under the null
    that both samples are drawn from the same underlying distribution (per problem),
    by simulating from the pooled distribution with the observed sample sizes.

    Returns: (results_by_problem, mean_js, std_js, p_value, z_score, null_mean, null_std)
    - results_by_problem: {problem_id: {'js_divergence': float, 'overlap': float, 'info': {...}}}
    - mean_js, std_js: across all problems (including empty-side problems, which use JSD=1.0)
    - p_value: Monte Carlo right-tailed p-value for the mean JSD across problems that have
               non-zero counts on both sides; None if no such problems.
    - z_score: standardized effect size for mean JSD: (observed_mean - null_mean)/null_std (None if undefined)
    - null_mean, null_std: mean and std of simulated mean JSD under the null (None if undefined)
    """
    problems = sorted(set(list(human_data.keys()) + list(llm_data.keys())))
    results = {}
    divergences_all = []

    # For Monte Carlo aggregation (only problems with both sides non-empty)
    paired_counts = []  # list of (h_counts:int[], l_counts:int[])
    js_nonempty = []

    for pid in problems:
        h = human_data.get(pid, {})
        l = llm_data.get(pid, {})
        o, info = js_divergence_for_task(h, l, use_fuzzy=use_fuzzy, fuzzy_threshold=fuzzy_threshold)
        # Provide both keys for backward compatibility; 'overlap' now equals JSD
        results[pid] = {'js_divergence': o, 'overlap': o, 'info': info}
        divergences_all.append(o)

        # Build aligned count arrays for testing
        _, h_counts, l_counts = _align_counts_for_problem(h, l, use_fuzzy=use_fuzzy, fuzzy_threshold=fuzzy_threshold)
        if int(np.sum(h_counts)) > 0 and int(np.sum(l_counts)) > 0:
            paired_counts.append((h_counts, l_counts))
            js_nonempty.append(_js_from_counts(h_counts, l_counts))

    mean = float(np.mean(divergences_all)) if divergences_all else 0.0
    std = float(np.std(divergences_all, ddof=0)) if divergences_all else 0.0

    # Monte Carlo p-value for the mean JSD across non-empty problems
    p_value = None
    z_score = None
    null_mean = None
    null_std = None
    if paired_counts:
        rng = np.random.default_rng(random_state)
        observed_mean = float(np.mean(js_nonempty))
        sim_means = []
        for _ in range(int(max(1, n_resamples))):
            total = 0.0
            count = 0
            for h_counts, l_counts in paired_counts:
                n_h = int(np.sum(h_counts))
                n_l = int(np.sum(l_counts))
                pooled = (h_counts + l_counts).astype(float)
                pooled_sum = float(np.sum(pooled))
                if pooled_sum <= 0:
                    continue
                prob = pooled / pooled_sum
                sim_h = rng.multinomial(n_h, prob)
                sim_l = rng.multinomial(n_l, prob)
                js = _js_from_counts(sim_h, sim_l)
                total += js
                count += 1
            if count > 0:
                sim_means.append(total / count)
        if sim_means:
            sims = np.asarray(sim_means, dtype=float)
            # Right-tailed p-value: proportion of simulated means >= observed
            p_value = float((np.sum(sims >= observed_mean) + 1.0) / (len(sims) + 1.0))
            null_mean = float(np.mean(sims))
            null_std = float(np.std(sims, ddof=0))
            if null_std > 0:
                z_score = float((observed_mean - null_mean) / null_std)

    return results, mean, std, p_value, z_score, null_mean, null_std


def plot_dataset_overlaps_from_stats(dataset_name, stats, methods_order=None, tasks_order=None,
                                     show=True, save_path=None, alpha: float = 0.05):
    """
    Draw a grouped bar plot for one dataset using pre-collected stats.
    stats: { task: { method_name: (mean, std), ... }, ... }
    """
    if methods_order is None:
        methods_order = ['vanilla', 'saliency', 'all-features', 'culture']
    if tasks_order is None:
        tasks_order = ['pick', 'guess', 'coordinate']

    n_tasks = len(tasks_order)
    n_methods = len(methods_order)

    means = np.zeros((n_tasks, n_methods))
    stds  = np.zeros((n_tasks, n_methods))
    pvals = np.full((n_tasks, n_methods), np.nan)
    zvals = np.full((n_tasks, n_methods), np.nan)

    for i, task in enumerate(tasks_order):
        for j, method in enumerate(methods_order):
            val = stats.get(task, {}).get(method)
            if val is None:
                means[i, j] = 0.0
                stds[i, j]  = 0.0
                pvals[i, j] = np.nan
            else:
                # Accept either (mean, std), (mean, std, p_value), or (mean, std, p_value, z)
                means[i, j] = float(val[0])
                stds[i, j]  = float(val[1])
                if len(val) >= 3 and val[2] is not None:
                    pvals[i, j] = float(val[2])
                if len(val) >= 4 and val[3] is not None:
                    zvals[i, j] = float(val[3])

    fig, ax = plt.subplots(figsize=(8, 5))
    group_x = np.arange(n_tasks)
    total_group_width = 0.75
    bar_width = total_group_width / n_methods
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * bar_width

    for j in range(n_methods):
        x_positions = group_x + offsets[j]
        bars = ax.bar(x_positions, means[:, j], width=bar_width,
                      yerr=stds[:, j], capsize=4, label=methods_order[j])
        # Outline bars with significant divergence
        for i, b in enumerate(bars):
            p = pvals[i, j]
            if not np.isnan(p) and p < alpha:
                b.set_linewidth(2.0)
                b.set_edgecolor('black')
            else:
                b.set_linewidth(0.0)

            # Add effect label near the top-right of each bar
            # Place above the error bar with a small margin, clipped to ylim
            y_top = float(means[i, j] + stds[i, j] + 0.02)
            y_top = min(y_top, 0.98)
            # Slightly to the right of the bar's center
            x_text = b.get_x() + b.get_width() * 0.60
            z = zvals[i, j]
            if not np.isnan(z):
                label = f"z={z:.2f}"
            else:
                # Fallback to p-value if z isn't available
                if np.isnan(p):
                    label = "p=n/a"
                else:
                    label = "p<0.001" if p < 0.001 else f"p={p:.3f}"
            ax.text(x_text, y_top, label, fontsize=8, ha='left', va='bottom')

    ax.set_xticks(group_x)
    ax.set_xticklabels(tasks_order, fontsize=11)
    ax.set_ylabel('Mean JS divergence', fontsize=11)
    ax.set_title(f'{dataset_name} — mean JS divergence per task (with std)', fontsize=12)
    ax.set_ylim(0.0, 1.0)
    ax.legend(title='Method', fontsize=9)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    return fig, ax


def plot_from_files(
    basepath, datasets, tasks, methods,
    load_human_data, load_llm_data, compute_overlaps_all_tasks,
    methods_order=None, tasks_order=None,
    show=True, save_dir=None, fuzzy_match=True,
    alpha: float = 0.05, n_resamples: int = 300, random_state: Optional[int] = None):
    """
    High-level function that:
      - Iterates datasets × tasks × methods (matching your main loop style)
      - Loads human & LLM data using your loader functions
      - Computes Jensen–Shannon divergences with compute_overlaps_all_tasks
      - Aggregates mean/std and a Monte Carlo p-value per dataset/task/method
        and plots per-dataset figures with significant bars outlined.
    
    Parameters
    ----------
    datasets : list[str]
    tasks : list[str]
    methods : dict[str, str]
        mapping method_name -> method_suffix (e.g. "saliency" -> "-instruct-saliency")
    load_human_data : callable(human_file_path, task) -> human_data_dict
    load_llm_data    : callable(llm_file_path, human_data) -> llm_data_dict
    compute_overlaps_all_tasks : callable(human_data, llm_data, **kwargs) -> (results, mean, std, p_value)
    methods_order : optional list specifying plot order (defaults to methods.keys() in insertion order)
    tasks_order : optional list specifying plot order (defaults to tasks)
    show : bool, whether to plt.show()
    save_dir : optional folder path to save each dataset figure (files named "<dataset>_overlap.png")
    fuzzy_match : unused here but kept for API compatibility if you need to pass to compute_overlaps
    """
    # Respect method order passed, otherwise preserve insertion order of methods dict
    if methods_order is None:
        methods_order = list(methods.keys())
    if tasks_order is None:
        tasks_order = list(tasks)

    # Container: stats_by_dataset[dataset][task][method] = (mean, std, p_value)
    stats_by_dataset = {ds: {t: {} for t in tasks_order} for ds in datasets}

    for dataset in datasets:
        for task in tasks:
            for method_name, method_suffix in methods.items():
                human_file_path = f'./data/Bardsley-humans/{dataset}_choices.json'
                llm_file_path = basepath + f'{dataset}{method_suffix}_problem-{task}.jsonl'

                # Load data (your loader functions must exist in the namespace)
                try:
                    human_data = load_human_data(human_file_path, task)
                except Exception as e:
                    print(f"[WARN] failed to load human data for {dataset}, {task}: {e}")
                    human_data = {}

                try:
                    llm_data = load_llm_data(llm_file_path, human_data)
                except Exception as e:
                    print(f"[WARN] failed to load llm data for {dataset}, {task}, method {method_name}: {e}")
                    llm_data = {}

                # Compute divergences and test metrics for mean JSD
                try:
                    _, mean, std, p_value, z_score, null_mean, null_std = compute_overlaps_all_tasks(
                        human_data, llm_data,
                        n_resamples=n_resamples, random_state=random_state,
                        use_fuzzy=fuzzy_match
                    )
                except Exception as e:
                    print(f"[WARN] compute_overlaps_all_tasks failed for {dataset},{task},{method_name}: {e}")
                    mean, std, p_value, z_score, null_mean, null_std = 0.0, 0.0, None, None, None, None

                # Save
                stats_by_dataset[dataset].setdefault(task, {})[method_name] = (mean, std, p_value, z_score)

                extra = f", p={p_value:.3f}" if p_value is not None else ""
                extra += f", z={z_score:.2f}" if z_score is not None else ""
                print(f"Dataset: {dataset}, Task: {task}, Method: {method_name} => Mean JS divergence: {mean:.4f} (std: {std:.4f}){extra}")
            print()

    # Now plot per dataset
    figs = {}
    for dataset in datasets:
        stats = stats_by_dataset[dataset]
        save_path = None
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{dataset}_overlaps.png")
        fig, ax = plot_dataset_overlaps_from_stats(
            dataset, stats, methods_order=methods_order, tasks_order=tasks_order,
            show=show, save_path=save_path, alpha=alpha
        )
        figs[dataset] = fig

    return stats_by_dataset, figs


def plot_ensemble_from_files(
    basepaths, datasets, tasks, methods,
    load_human_data, load_llm_data, compute_overlaps_all_tasks,
    methods_order=None, tasks_order=None,
    show=True, save_dir: Optional[str] = "./results/human_llm_coordination/llms-ensemble/", fuzzy_match=True,
    alpha: float = 0.05, n_resamples: int = 300, random_state: Optional[int] = None
):
    """
    Ensemble across models per method, per dataset and task:
    - For each dataset and task, and for each method in methods_order,
      aggregate LLM counts across all basepaths (models) for every problem.
    - Compare the aggregated model distribution to humans (per problem),
      compute JSD stats and Monte Carlo test, and plot bars per method (like a single-model plot).

    Saves one figure per dataset. Returns figures keyed by dataset.
    """
    if methods_order is None:
        methods_order = list(methods.keys())
    if tasks_order is None:
        tasks_order = list(tasks)

    # Helper to aggregate LLM counts across all models for a given dataset/method/task
    def _aggregate_llm_across_models(dataset, task, method_suffix, human_data):
        combined = {pid: {} for pid in human_data.keys()}
        for basepath in basepaths:
            file_path = basepath + f"{dataset}{method_suffix}_problem-{task}.jsonl"
            try:
                llm_data = load_llm_data(file_path, human_data)
            except Exception:
                llm_data = {}
            for pid, resp_counts in (llm_data or {}).items():
                bucket = combined.setdefault(pid, {})
                for k, v in (resp_counts or {}).items():
                    bucket[k] = bucket.get(k, 0) + int(v)
        return combined

    # Build stats: per dataset, per task, one bar per method (aggregated across models)
    stats_by_dataset = {ds: {t: {} for t in tasks_order} for ds in datasets}

    for dataset in datasets:
        for task in tasks_order:
            human_file_path = f'./data/Bardsley-humans/{dataset}_choices.json'
            try:
                human_data = load_human_data(human_file_path, task)
            except Exception as e:
                print(f"[WARN] failed to load human data for {dataset}, {task}: {e}")
                human_data = {}

            for method_name, method_suffix in methods.items():
                aggregated_llm = _aggregate_llm_across_models(dataset, task, method_suffix, human_data)
                try:
                    _, mean, std, p_value, z_score, null_mean, null_std = compute_overlaps_all_tasks(
                        human_data, aggregated_llm,
                        n_resamples=n_resamples, random_state=random_state,
                        use_fuzzy=fuzzy_match
                    )
                except Exception as e:
                    print(f"[WARN] compute_overlaps_all_tasks failed for ensemble {dataset}, {task}, method {method_name}: {e}")
                    mean, std, p_value, z_score = 0.0, 0.0, None, None

                stats_by_dataset[dataset].setdefault(task, {})[method_name] = (
                    mean, std, p_value, z_score
                )

    # Plot per dataset with one bar per method (aggregated across models)
    figs = {}
    import os
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for dataset in datasets:
        task_stats = stats_by_dataset[dataset]
        save_path = os.path.join(save_dir, f"{dataset}_overlaps.png") if save_dir else None
        fig, ax = plot_dataset_overlaps_from_stats(
            dataset, task_stats, methods_order=methods_order, tasks_order=tasks_order,
            show=show, save_path=save_path, alpha=alpha
        )
        if save_path:
            print(f"[INFO] Saved ensemble figure: {save_path}")
        figs[dataset] = fig

    return figs



if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser(description="Human–LLM coordination analysis and plotting")
    parser.add_argument("--ensemble", action="store_true",
                        help="If set, combine all models into a single plot per dataset and save under ./plots/llms-ensemble/")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for outlines")
    parser.add_argument("--n-resamples", type=int, default=300, help="Monte Carlo resamples for p-value/z-score")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for Monte Carlo")
    parser.add_argument("--no-show", action="store_true", help="Do not display plots interactively")
    args = parser.parse_args()

    basepaths = [
        "./results/meta-llama/Llama-3.1-8B-Instruct/",
        "./results/meta-llama/Meta-Llama-3-8B-Instruct/",
        "./results/meta-llama/Llama-3.2-3B-Instruct/",
        "./results/meta-llama/Llama-3.2-1B-Instruct/",
        './results/meta-llama/Llama-3.3-70B-Instruct/',
        './results/meta-llama/Llama-3.1-70B-Instruct/',
        './results/meta-llama/Meta-Llama-3-70B-Instruct/',
        './results/Qwen/Qwen2.5-72B-Instruct/',
        './results/Qwen/Qwen2-72B-Instruct/',
        "./results/Qwen/Qwen2.5-32B-Instruct/",
        "./results/Qwen/Qwen2.5-14B-Instruct-1M/",
        "./results/Qwen/Qwen2.5-14B-Instruct/",
        "./results/Qwen/Qwen2.5-7B-Instruct-1M/",
        "./results/Qwen/Qwen2.5-7B-Instruct/",
        "./results/Qwen/Qwen2-7B-Instruct/",
        "./results/Qwen/Qwen2.5-3B-Instruct/",
        "./results/Qwen/Qwen2.5-1.5B-Instruct/",
        "./results/Qwen/Qwen2-1.5B-Instruct/",
        "./results/Qwen/Qwen2.5-0.5B-Instruct/",
        "./results/Qwen/Qwen2-0.5B-Instruct/"
                ]
    datasets = ["amsterdam", "nottingham"]
    tasks = ["pick", "guess", "coordinate"]
    methods = {
        "vanilla": "",
        "saliency": "-instruct-saliency",
        "all-features": "-instruct-all-features",
        # "culture": "-instruct-culture",  # add this when we have the result
    }

    show = not args.no_show

    if args.ensemble:
        # One combined plot per dataset across all models
        _ = plot_ensemble_from_files(
            basepaths, datasets, tasks, methods,
            load_human_data, load_llm_data, compute_overlaps_all_tasks,
            methods_order=list(methods.keys()),
            tasks_order=list(tasks),
            show=show, save_dir="./plots/human_llm_coordination/llms-ensemble/", fuzzy_match=True,
            alpha=args.alpha, n_resamples=args.n_resamples, random_state=args.random_state
        )
    else:
        # Separate plots per model × dataset
        for basepath in basepaths:
            base_model = basepath.split('/')[-2].lower()
            _stats_by_dataset, _figs = plot_from_files(
                basepath, datasets, tasks, methods,
                load_human_data, load_llm_data, compute_overlaps_all_tasks,
                methods_order=list(methods.keys()),
                tasks_order=list(tasks),
                show=show,
                save_dir=f'./plots/human_llm_coordination/{base_model}/',
                alpha=args.alpha,
                n_resamples=args.n_resamples,
                random_state=args.random_state
            )
                
                
