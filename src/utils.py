import json
import itertools
import matplotlib.pyplot as plt
import re
import textwrap
from collections import Counter


# Plot function
def plot_block_frequencies(data, dataset, model_name, problem_tag):
    jsonl_results = []
    for block in data:
        responses = []
        for r in block["responses"]:
            if r is not None:
                responses.append(r)
        count = Counter(responses)
        print(count)

        if count:
            # Coordination Index
            total = sum(count.values())
            if total > 1:
                coordination_index = sum([v * (v - 1) for v in dict(count).values()]) / (total * (total - 1))
                normalised_coordination_index = coordination_index * total / len(count)
                print(f"Coordination Index: {coordination_index}")
                print(f"Normalised Coordination Index: {normalised_coordination_index}")
            else:
                coordination_index = 0

            # Create the figure with two subplots side by side
            fig, (ax_bar, ax_text) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})

            # Plot the bar chart on the left
            ax_bar.bar(count.keys(), count.values())
            ax_bar.set_title(f"Frequencies for prompt with idx {block['idx']}.{block['variation-idx']}")
            ax_bar.set_xlabel("Answer")
            ax_bar.set_ylabel("Frequency")
            ax_bar.tick_params(axis='x', rotation=45)

            # Add the wrapped prompt text on the right
            max_line_width = 45  # characters
            wrapped_text = "\n".join(textwrap.wrap(block['prompt'], width=max_line_width))
            ax_text.text(0, 1, wrapped_text, fontsize=12, va='top',
                        bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'))
            ax_text.axis('off')  # Hide axis for the text box

            plt.tight_layout()
            print(block['idx'])

            # Save the figure
            plt.savefig(
                f"./images/{model_name}/{dataset}/{problem_tag}/idx_{block['idx']}.{block['variation-idx']}_frequencies.png",
                bbox_inches='tight',
                dpi=150)

            jsonl_results.append({
                "idx": block["idx"],
                "variation-idx": block['variation-idx'],
                "prompt": block["prompt"],
                "responses": dict(count),
                "coordination_index": coordination_index,
                "normalised_coordination_index": normalised_coordination_index
            })

        else:
            print(f"No valid responses found for block with idx {block['idx']} and prompt `{block['prompt']}'.")

        print("=" * 80)
        
    with open(f"./results/{model_name}/{dataset}_{problem_tag}.jsonl", "w") as f:
        json.dump(jsonl_results, f, indent=2)


def iterate_data(data: dict, problem_tag: str):
    problems = {}
    for d in data:
        idx = d["id"]
        problems[idx] = []
        text = d[problem_tag]
        placeholders = [d[p] for p in d["placeholders"]]
        if not placeholders:
            problems[idx].append(text)
            continue

        for elements in itertools.product(*placeholders):
            new_text = text
            assert len(elements) == len(d["placeholders"]), "Mismatch in number of placeholders and elements"
            for el, p in zip(elements, d["placeholders"]):
                new_text = new_text.replace(f"@{p}@", el)
            problems[idx].append(new_text)

    return problems


# def coordination_index(items: dict):
# TODO: Integrate this function with metrics.py
#     freq_counter = Counter(items)
#     total = sum(freq_counter.values())
#     coordination_index = sum([v * (v - 1) for v in dict(freq_counter).values()]) / (total * (total - 1))
#     normalised_coordination_index = coordination_index * total / len(freq_counter)
    
#     return coordination_index, normalised_coordination_index
