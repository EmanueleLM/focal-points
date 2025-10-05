import glob
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import re
from collections import defaultdict
from pathlib import Path

def sample(d: dict, n: int, replacement: bool = True) -> dict:
    
    if replacement:
        if not d or n >= sum(d.values()):
            return d
    
    # Create the full population list
    population = []
    for key, count in d.items():
        population.extend([key] * count)
    
    # Adjust n for no-replacement
    if not replacement and n > len(population):
        n = len(population)
    
    # Sample from the population
    samples = np.random.choice(population, size=n, replace=replacement)
    
    # Count occurrences
    result = {}
    for s in samples:
        result[str(s)] = result.get(s, 0) + 1
    
    return result

def get_family(model_name):
    return model_name.split("/")[1] if model_name.startswith("/") else model_name.split("/")[0]

def get_color(model_name, models_in_family):
    family = get_family(model_name)
    cmap = family_colormaps[family]
    print(model_name, models_in_family)
    idx = models_in_family.index(model_name)
    n = len(models_in_family)
    return cmap(0.3 + 0.6 * idx / (n - 1 if n > 1 else 1))  # spaced shades

from pathlib import Path

def get_available_models(model_names, required_files):
    base = Path("./results")
    working_model_names = []
    corrupted_model_names = {}
    required = set(required_files)

    for m in model_names:
        # Ignore entries like ".DS_Store" that slipped into model_names
        if Path(m).name.startswith("."):
            continue

        model_dir = base / m.lstrip("/")  # handle leading slash in m
        # Collect only real, non-hidden files
        files = {p.name for p in model_dir.glob("*") if p.is_file() and not p.name.startswith(".")}

        missing_files = required - files
        if missing_files:
            corrupted_model_names[m] = missing_files
        else:
            working_model_names.append(m)

    if corrupted_model_names:
        print("\n--- Corrupted Models (missing files) ---")
        for m, missing in corrupted_model_names.items():
            print(f"Model: {m}")
            print(f"Missing files: {missing}")
            print()
        raise ValueError("Some models are corrupted. See above for details.")

    return working_model_names


# Define families and assign colormaps
family_colormaps = defaultdict(
    lambda: cm.Greys,  # default colormap
    {
        "meta-llama": cm.Blues,
        "google": cm.Reds,
        "microsoft": cm.Greens,
        "Qwen": cm.Purples,
        "deepseek-ai": cm.Oranges,
    }
)

# Datasets and tasks (we name them labels)
dataset_names = ["amsterdam", "nottingham"]
labels = ["pick", "guess", "coordinate"]

required_files = [
    "amsterdam_numeric-instruct-all-features_problem-coordinate.jsonl",
    "amsterdam_numeric-instruct-all-features_problem-guess.jsonl",
    "amsterdam_numeric-instruct-all-features_problem-pick.jsonl",
    "nottingham_numeric-instruct-all-features_problem-coordinate.jsonl",
    "nottingham_numeric-instruct-all-features_problem-guess.jsonl",
    "nottingham_numeric-instruct-all-features_problem-pick.jsonl"
]

# Families of models we test: each corresponds to a folder in ./results
models = ["Qwen"]
num_samples = 1000
sample_with_replacement = True
mode_variations = [("-all-variations", 3)]

# Matplotlib config
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'

# Create save directories
SAVEDIR = "./plots"
sample_folder = ('sample-with-replacement' if sample_with_replacement else 'sample-without-replacement')

COORDINATION_INDEX_FOLDER = f"/all-features-numeric/coordination-index/{sample_folder}/{models[0]}"
BEST_MODELS_COORDINATION_INDEX_FOLDER = f"/all-features-numeric/coordination-index-best/{sample_folder}/{models[0]}"
BEST_MODELS_COORDINATION_INDEX_SAMPLING_FOLDER = f"/all-features-numeric/coordination-index-best-merge/{sample_folder}/{models[0]}"

for folder in [COORDINATION_INDEX_FOLDER, 
               BEST_MODELS_COORDINATION_INDEX_FOLDER,
               BEST_MODELS_COORDINATION_INDEX_SAMPLING_FOLDER
               ]:
    if not os.path.exists(Path(SAVEDIR + folder)):
        print("Creating path: ", Path(SAVEDIR + folder))
        os.makedirs(Path(SAVEDIR + folder))
    
if __name__ == "__main__":

    # Collect the names of the folders inside each "model"
    model_names = []
    for m in models:
        for model_path in next(os.walk("./results/" + m)):
            if "./results/" in model_path:
                prefix = model_path
                continue
            model_names.extend([prefix.split("./results")[1] + "/" + llm for llm in model_path])

    print("--- Available Models ---")
    for i,m in enumerate(model_names):
        print(f"{i+1}. {m}")
        
    model_names = get_available_models(model_names, required_files)
    
    # Sort by model-size
    model_by_size = {}
    for m in model_names:
        match = re.search(r"-([0-9]+)([B])(?:-|$)", m, re.DOTALL)
        size = int(match.group(1).strip())
        model_by_size[m] = size
    
    model_by_size = {k: v for k, v in sorted(model_by_size.items(), key=lambda item: item[1])}
    model_names = list(model_by_size.keys())

    family_models = defaultdict(list)
    for m in model_names:
        family_models[get_family(m)].append(m)
    
    # 1. Coordination Index -- All Llamas
    for suffix, mega_i in mode_variations:    
        for d_name in dataset_names:
            print(f"Dataset: {d_name}")
            
            # Load the human results (we need them for the normalisation factor)
            with open(f"./data/{d_name}.jsonl", "r") as f:
                data_humans = json.load(f)
            normalization_factors = [d["normalization_factor"] for d in data_humans]
    
            for l in labels:
                print(f"Label: {l}")
                # Load LLM results
                data_llms = {}
                for model_name in model_names:
                    print(model_name)
                    with open(f"./results{model_name}/{d_name}_problem-{l}.jsonl", "r") as f:
                        data_llm = json.load(f)
                    
                    data_llms[model_name] = []
                    for i in range(0, len(data_llm), 3):
                        current_responses = {}
                        for d in data_llm[i:i+mega_i]:
                            # For each task, put together the tasks and then average
                            for response in d["responses"].keys():
                                match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

                                if match:
                                    re_response = match.group(1).strip().lower()
                                    if re_response not in current_responses:
                                        current_responses[re_response] = 0
                                    current_responses[re_response] += d["responses"][response]
                        
                        # Normalised coordination index
                        current_responses = sample(current_responses, n=num_samples, replacement=sample_with_replacement)
                        N = sum(list(current_responses.values()))
                        n = normalization_factors[i%3]
                        nci = 0.
                        if N > 1:
                            for _,v in current_responses.items():
                                nci += v*(v-1)
                            data_llms[model_name].append(nci*n/(N*(N-1)))
                        else:
                            data_llms[model_name].append(0.)
                            
                # Load Human results
                with open(f"./data/Bardsley-humans/{d_name}.jsonl", "r") as f:
                    results_humans = json.load(f)
                current_data_humans = [
                    d["normalised_coordination_index"] for d in results_humans if d["task"] == l
                ]

                # ---- Plot ----
                plt.figure(figsize=(12, 5))

                tasks = [f"T{d_name[0].upper()}{i+1}" for i in range(14)]
                x = np.arange(len(tasks))  # label locations
                width = 0.1  # width of each bar

                # Plot human data
                plt.bar(x - width, current_data_humans, width, label="Humans", color="black")

                # Plot LLM data
                for idx, model_name in enumerate(model_names):
                    current_data_llm = data_llms[model_name]
                    family = get_family(model_name)
                    color = get_color(model_name, family_models[family])
                    plt.bar(x + width*(idx), current_data_llm, width, label=model_name, color=color, edgecolor="black")

                plt.xticks(x, tasks)
                plt.title(f"Coordination Index Comparison: {d_name.capitalize()}-{l}")
                plt.ylabel("Normalised Coordination Index")
                plt.legend(bbox_to_anchor=(1., 1.))
                plt.tight_layout()
                plt.grid(axis='y', alpha=0.3)
                plt.savefig(SAVEDIR + COORDINATION_INDEX_FOLDER + f"/{d_name}-{l}{suffix}.png")
                plt.show()
            
    # 2. Coordination Index -- Best llamas
    model_names = [
        "/Qwen/Qwen2-72B-Instruct",
        "/Qwen/Qwen2.5-72B-Instruct",
        ]
        
    for suffix, mega_i in mode_variations:
        for d_name in dataset_names:
            print(f"Dataset: {d_name}")
            
            # Load the human results (we need them for the normalisation factor)
            with open(f"./data/{d_name}.jsonl", "r") as f:
                data_humans = json.load(f)
            normalization_factors = [d["normalization_factor"] for d in data_humans]
    
            for l in labels:
                print(f"Label: {l}")
                # Load LLM results
                data_llms = {}
                for model_name in model_names:
                    print(model_name)
                    with open(f"./results{model_name}/{d_name}_problem-{l}.jsonl", "r") as f:
                        data_llm = json.load(f)
                    
                    data_llms[model_name] = []
                    for i in range(0, len(data_llm), 3):
                        current_responses = {}
                        for d in data_llm[i:i+mega_i]:
                            # For each task, put together the tasks and then average
                            for response in d["responses"].keys():
                                match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

                                if match:
                                    re_response = match.group(1).strip().lower()
                                    if re_response not in current_responses:
                                        current_responses[re_response] = 0
                                    current_responses[re_response] += d["responses"][response]
                        
                        # Normalised coordination index
                        current_responses = sample(current_responses, n=num_samples, replacement=sample_with_replacement)
                        N = sum(list(current_responses.values()))
                        n = normalization_factors[i%3]
                        nci = 0.
                        if N > 1:
                            for _,v in current_responses.items():
                                nci += v*(v-1)
                            data_llms[model_name].append(nci*n/(N*(N-1)))
                        else:
                            data_llms[model_name].append(0.)
                            
                # Load Human results
                with open(f"./data/Bardsley-humans/{d_name}.jsonl", "r") as f:
                    results_humans = json.load(f)
                current_data_humans = [
                    d["normalised_coordination_index"] for d in results_humans if d["task"] == l
                ]

                # ---- Plot ----
                plt.figure(figsize=(12, 5))

                tasks = [f"T{d_name[0].upper()}{i+1}" for i in range(14)]
                x = np.arange(len(tasks))  # label locations
                width = 0.1  # width of each bar

                # Plot human data
                plt.bar(x - width, current_data_humans, width, label="Humans", color="black")

                # Plot LLM data
                for idx, model_name in enumerate(model_names):
                    current_data_llm = data_llms[model_name]
                    family = get_family(model_name)
                    color = get_color(model_name, family_models[family])
                    plt.bar(x + width*(idx), current_data_llm, width, label=model_name, color=color, edgecolor="black")

                plt.xticks(x, tasks)
                plt.title(f"Coordination Index Comparison: {d_name.capitalize()}-{l}")
                plt.ylabel("Normalised Coordination Index")
                plt.legend(bbox_to_anchor=(1., 1.))
                plt.tight_layout()
                plt.grid(axis='y', alpha=0.3)
                plt.savefig(SAVEDIR + BEST_MODELS_COORDINATION_INDEX_FOLDER + f"/{d_name}-{l}{suffix}.png")
                plt.show()
            
    # 3. Coordination Index -- Merge best llamas
    model_names = [
        "/Qwen/Qwen2-72B-Instruct",
        "/Qwen/Qwen2.5-72B-Instruct",
        ]
    
    for suffix, mega_i in mode_variations:  
        for d_name in dataset_names:
            print(f"Dataset: {d_name}")
            # Load the human results (we need them for the normalisation factor)
            with open(f"./data/{d_name}.jsonl", "r") as f:
                data_humans = json.load(f)
            normalization_factors = [d["normalization_factor"] for d in data_humans]
            
            for l in labels:
                data_llms = []
                print(f"Label: {l}")

                # Load LLM results
                for i in range(0, len(data_humans)*3, 3):  # TA
                    current_responses = {}
                    for model_name in model_names:
                        with open(f"./results{model_name}/{d_name}_problem-{l}.jsonl", "r") as f:
                            data_llm = json.load(f)
                        
                        for d in data_llm[i:i+mega_i]:
                            # For each task, put together the tasks and then average
                            for response in d["responses"].keys():
                                match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

                                if match:
                                    re_response = match.group(1).strip().lower()
                                    if re_response not in current_responses:
                                        current_responses[re_response] = 0
                                    current_responses[re_response] += d["responses"][response]
                        
                    # Normalised coordination index
                    current_responses = sample(current_responses, n=num_samples, replacement=sample_with_replacement)
                    N = sum(list(current_responses.values()))
                    n = normalization_factors[i%3]
                    nci = 0.
                    if N > 1:
                        for _,v in current_responses.items():
                            nci += v*(v-1)
                        data_llms.append(nci*n/(N*(N-1)))
                    else:
                        data_llms.append(0.)

                # Load Human results
                with open(f"./data/Bardsley-humans/{d_name}.jsonl", "r") as f:
                    results_humans = json.load(f)
                current_data_humans = [
                    dd["normalised_coordination_index"] for dd in results_humans if dd["task"] == l
                ]

                # ---- Plot ----
                plt.figure(figsize=(12, 5))

                tasks = [f"T{d_name[0].upper()}{i+1}" for i in range(14)]
                x = np.arange(len(tasks))  # label locations
                width = 0.3  # width of each bar

                # Plot human data (shifted left)
                human_bars = plt.bar(x - width/2, current_data_humans, width, label="Humans", color="black")

                # Plot meta-llama data (shifted right)
                llm_bars = plt.bar(x + width/2, data_llms, width, label="meta-llama", color="blue", edgecolor="black")

                # Add values on top of human bars
                for bar in human_bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.2f}', ha='center', va='bottom', fontsize=8)

                # Add values on top of LLM bars
                for bar in llm_bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.2f}', ha='center', va='bottom', fontsize=8)

                plt.xticks(x, tasks)
                plt.title(f"Coordination Index Comparison: {d_name.capitalize()}-{l}")
                plt.ylabel("Normalised Coordination Index")
                plt.legend(bbox_to_anchor=(1., 1.))
                plt.tight_layout()
                plt.grid(axis='y', alpha=0.3)
                plt.savefig(SAVEDIR + BEST_MODELS_COORDINATION_INDEX_SAMPLING_FOLDER + f"/{d_name}-{l}{suffix}.png")
                plt.show()
            