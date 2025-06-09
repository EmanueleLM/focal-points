import json
import itertools
import matplotlib.pyplot as plt
import re
import textwrap
from collections import Counter


# Plot function
def plot_block_frequencies(data, dataset, model_name, problem_tag):
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
                print(f"Coordination Index: {coordination_index:.4f}")
                print(f"Normalised Coordination Index: {coordination_index * len(count):.4f}")
            else:
                coordination_index = 0

            # Plotting
            plt.figure(figsize=(10, 5))
            plt.bar(count.keys(), count.values())
            plt.title(f"Frequencies for prompt with idx {block['idx']}.{block['variation-idx']}")
            plt.xlabel("Answer")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45, ha="right")

            # Add the prompt text as a text box
            max_line_width = 45  # characters
            wrapped_text = "\n".join(textwrap.wrap(block['prompt'], width=max_line_width))
            plt.text(0.5, 0.5, wrapped_text,
                     fontsize=12,
                     bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'))

            plt.tight_layout()
            print(block['idx'])
            plt.savefig(
                f"./images/{model_name}/{dataset}/{problem_tag}/idx_{block['idx']}.{block['variation-idx']}_frequencies.png",
                bbox_inches='tight',
                dpi=150)

            with open(f"./results/{model_name}/{dataset}_{problem_tag}.jsonl", "a") as f:
                f.write(json.dumps({
                    "idx": block["idx"],
                    "variation-idx": block['variation-idx'],
                    "prompt": block["prompt"],
                    "responses": dict(count),
                    "coordination_index": coordination_index,
                    "normalised_coordination_index": coordination_index * len(count)
                }, indent=2) + "\n")

        else:
            print(f"No valid responses found for block with idx {block['idx']} and prompt `{block['prompt']}'.")

        print("=" * 80)


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


def coordination_index(items: dict):
    freq = freq_counter = Counter(items)
    total = sum(freq_counter.values())
    coordination_index = sum([v * (v - 1) for v in dict(freq).values()]) / (total * (total - 1))

    return coordination_index, coordination_index * len(freq_counter)
