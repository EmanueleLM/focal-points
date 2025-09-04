import glob
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import re
from collections import defaultdict
from pathlib import Path


def sample(dict: dict, n: int) -> dict:
    # Keys and weights
    elements = np.array(list(dict.keys()))
    weights = np.array(list(dict.values()), dtype=float)

    # Normalize weights to probabilities
    probabilities = weights / weights.sum()

    # Sample with replacement
    samples = np.random.choice(elements, size=n, p=probabilities)

    # Count occurrences and convert back to dictionary
    sampled_dict = {
        str(key): int(count) for key, count in zip(*np.unique(samples, return_counts=True))
    }

    return sampled_dict

def get_family(model_name):
    return model_name.split("/")[1] if model_name.startswith("/") else model_name.split("/")[0]

def get_color(model_name, models_in_family):
    family = get_family(model_name)
    cmap = family_colormaps[family]
    idx = models_in_family.index(model_name)
    n = len(models_in_family)
    return cmap(0.3 + 0.6 * idx / (n - 1 if n > 1 else 1))  # spaced shades

def get_available_models(model_names, required_files):
    working_model_names = []
    corrupted_model_names = {}
    required = set(required_files)
    for m in model_names:
        files = set(glob.glob(f"./results{m}/*"))
        files = set([f.split('/')[-1] for f in files])
        missing_files = required.difference(files)
        if missing_files != set():
            corrupted_model_names[m] = required.difference(files)
        else:
            working_model_names.append(m)
            
    return working_model_names

# Define families and assign colormaps
family_colormaps = {
    "meta-llama": cm.Blues,
    "google": cm.Reds,
    "microsoft": cm.Greens,
    "Qwen": cm.Purples,
    "deepseek-ai": cm.Oranges,
}

# Files required to run all the experiments
required_files = [
    "amsterdam_problem-coordinate.jsonl",
    "amsterdam_problem-guess.jsonl",
    "amsterdam_problem-pick.jsonl",
    "nottingham_problem-coordinate.jsonl",
    "nottingham_problem-guess.jsonl",
    "nottingham_problem-pick.jsonl"
]

# Families of models we test: each corresponds to a folder in ./results
models = ["meta-llama"]


# Datasets and tasks (we name them labels)
dataset_names = ["amsterdam", "nottingham"]
labels = ["pick", "guess", "coordinate"]

# Matplotlib config
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'

# Savedir (creation)
SAVEDIR = "./plots"
COORDINATION_INDEX_FOLDER = "/coordination-index/llama"
BEST_MODELS_COORDINATION_INDEX_FOLDER = "/best-models-coordination-index/llama"
BEST_MODELS_COORDINATION_INDEX_SAMPLING_FOLDER = "/best-models-coordination-index-sampling/llama"

for folder in [COORDINATION_INDEX_FOLDER, 
               BEST_MODELS_COORDINATION_INDEX_FOLDER,
               BEST_MODELS_COORDINATION_INDEX_SAMPLING_FOLDER]:
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
    
    # 1. Coordination Index -- All Llamas
    family_models = defaultdict(list)
    for m in model_names:
        family_models[get_family(m)].append(m)
        
    for d_name in dataset_names:
        print(f"Dataset: {d_name}")
        for l in labels:
            print(f"Label: {l}")

            # Load LLM results
            data_llms = {}
            for model_name in model_names:
                print(model_name)
                with open(f"./results{model_name}/{d_name}_problem-{l}.jsonl", "r") as f:
                    data_llm = json.load(f)
                    
                    
                data_llms[model_name] = [
                    sum([d["normalised_coordination_index"] for d in data_llm[i:i+3]])/3
                    for i in range(0, len(data_llm), 3)
                ]
                
                data_llms[model_name] = []
                for i in range(0, len(data_llm), 3):
                    current_responses = {}
                    for d in data_llm[i:i+3]:
                        # For each task, put together the tasks and then average
                        for response in d["responses"].keys():
                            match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

                            if match:
                                re_response = match.group(1).strip()
                                if re_response not in current_responses:
                                    current_responses[re_response] = 0
                                current_responses[re_response] += d["responses"][response]
                            else:
                                pass
                    
                    # Normalised coordination index
                    nci = 0.
                    N = sum(list(current_responses.values()))
                    n = len(current_responses.keys())
                    for _,v in current_responses.items():
                        nci += v*(v-1)
                    
                    if N > 1:
                        data_llms[model_name].append(nci*n/(N*(N-1)))
                    else:
                        data_llms[model_name].append(0.)

            # Load Human results
            with open(f"./data/Bardsley-humans/{d_name}.jsonl", "r") as f:
                data_humans = json.load(f)
            current_data_humans = [
                d["normalised_coordination_index"] for d in data_humans if d["task"] == l
            ]

            # ---- Plot ----
            plt.figure(figsize=(12, 5))
            plt.plot(current_data_humans, label="Humans", marker="o", color="black")

            for model_name in model_names:
                current_data_llm = data_llms[model_name]
                family = get_family(model_name)
                color = get_color(model_name, family_models[family])
                plt.plot(current_data_llm, label=model_name, marker="x", color=color)

            plt.xticks([i for i in range(14)], [f"T{d_name[0].upper()}{i+1}" for i in range(14)])
            plt.title(f"Coordination Index Comparison: {d_name.capitalize()}-{l}")
            plt.ylabel("Normalised Coordination Index")
            plt.legend(bbox_to_anchor=(1., 1.))
            plt.tight_layout()
            plt.grid()
            plt.savefig(SAVEDIR + COORDINATION_INDEX_FOLDER + f"/{d_name}-{l}.png")
            
    # 2. Coordination Index -- Best llamas
    model_names = ["/meta-llama/Llama-3.1-70B-Instruct",
                   "/meta-llama/Llama-3.3-70B-Instruct",
                   "/meta-llama/Llama-3.3-70B-Instruct",
                   ]
    
    family_models = defaultdict(list)
    for m in model_names:
        family_models[get_family(m)].append(m)
        
    for d_name in dataset_names:
        print(f"Dataset: {d_name}")
        for l in labels:
            print(f"Label: {l}")

            # Load LLM results
            data_llms = {}
            for model_name in model_names:
                print(model_name)
                with open(f"./results{model_name}/{d_name}_problem-{l}.jsonl", "r") as f:
                    data_llm = json.load(f)
                    
                    
                data_llms[model_name] = [
                    sum([d["normalised_coordination_index"] for d in data_llm[i:i+3]])/3
                    for i in range(0, len(data_llm), 3)
                ]
                
                data_llms[model_name] = []
                for i in range(0, len(data_llm), 3):
                    current_responses = {}
                    for d in data_llm[i:i+3]:
                        # For each task, put together the tasks and then average
                        for response in d["responses"].keys():
                            match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

                            if match:
                                re_response = match.group(1).strip()
                                if re_response not in current_responses:
                                    current_responses[re_response] = 0
                                current_responses[re_response] += d["responses"][response]
                            else:
                                pass
                    
                    # Normalised coordination index
                    nci = 0.
                    N = sum(list(current_responses.values()))
                    n = len(current_responses.keys())
                    for _,v in current_responses.items():
                        nci += v*(v-1)
                    
                    if N > 1:
                        data_llms[model_name].append(nci*n/(N*(N-1)))
                    else:
                        data_llms[model_name].append(0.)

            # Load Human results
            with open(f"./data/Bardsley-humans/{d_name}.jsonl", "r") as f:
                data_humans = json.load(f)
            current_data_humans = [
                d["normalised_coordination_index"] for d in data_humans if d["task"] == l
            ]

            # ---- Plot ----
            plt.figure(figsize=(12, 5))
            plt.plot(current_data_humans, label="Humans", marker="o", color="black")

            for model_name in model_names:
                current_data_llm = data_llms[model_name]
                family = get_family(model_name)
                color = get_color(model_name, family_models[family])
                plt.plot(current_data_llm, label=model_name, marker="x", color=color)

            plt.xticks([i for i in range(14)], [f"T{d_name[0].upper()}{i+1}" for i in range(14)])
            plt.title(f"Coordination Index Comparison: {d_name.capitalize()}-{l}")
            plt.ylabel("Normalised Coordination Index")
            plt.legend(bbox_to_anchor=(1., 1.))
            plt.tight_layout()
            plt.grid()
            plt.savefig(SAVEDIR + BEST_MODELS_COORDINATION_INDEX_FOLDER + f"/{d_name}-{l}.png")
            
    # 3. Coordination Index -- Best llamas but this time we merge and then sample
    model_names = ["/meta-llama/Llama-3.1-70B-Instruct",
                   "/meta-llama/Llama-3.3-70B-Instruct",
                   "/meta-llama/Llama-3.3-70B-Instruct",
                   ]
    
    family_models = defaultdict(list)
    for m in model_names:
        family_models[get_family(m)].append(m)
        
    for d_name in dataset_names:
        print(f"Dataset: {d_name}")
        for l in labels:
            print(f"Label: {l}")

            # Load LLM results
            data_llms = {}
            for model_name in model_names:
                print(model_name)
                with open(f"./results{model_name}/{d_name}_problem-{l}.jsonl", "r") as f:
                    data_llm = json.load(f)
                    
                    
                data_llms["meta-llama"] = [
                    sum([d["normalised_coordination_index"] for d in data_llm[i:i+3]])/3
                    for i in range(0, len(data_llm), 3)
                ]
                
                data_llms["meta-llama"] = []
                for i in range(0, len(data_llm), 3):
                    current_responses = {}
                    for d in data_llm[i:i+3]:
                        # For each task, put together the tasks and then average
                        for response in d["responses"].keys():
                            match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

                            if match:
                                re_response = match.group(1).strip()
                                if re_response not in current_responses:
                                    current_responses[re_response] = 0
                                current_responses[re_response] += d["responses"][response]
                            else:
                                pass
                    
                    # Normalised coordination index
                    current_responses = sample(current_responses, n=64)
                    nci = 0.
                    N = sum(list(current_responses.values()))
                    n = len(current_responses.keys())
                    for _,v in current_responses.items():
                        nci += v*(v-1)
                    
                    if N > 1:
                        data_llms["meta-llama"].append(nci*n/(N*(N-1)))
                    else:
                        data_llms["meta-llama"].append(0.)

            # Load Human results
            with open(f"./data/Bardsley-humans/{d_name}.jsonl", "r") as f:
                data_humans = json.load(f)
            current_data_humans = [
                d["normalised_coordination_index"] for d in data_humans if d["task"] == l
            ]

            # ---- Plot ----
            plt.figure(figsize=(12, 5))
            plt.plot(current_data_humans, label="Humans", marker="o", color="black")

            current_data_llm = data_llms["meta-llama"]
            family = get_family("meta-llama")
            color = "blue"
            plt.plot(current_data_llm, label="meta-llama", marker="x", color=color)

            plt.xticks([i for i in range(14)], [f"T{d_name[0].upper()}{i+1}" for i in range(14)])
            plt.title(f"Coordination Index Comparison: {d_name.capitalize()}-{l}")
            plt.ylabel("Normalised Coordination Index")
            plt.legend(bbox_to_anchor=(1., 1.))
            plt.tight_layout()
            plt.grid()
            plt.savefig(SAVEDIR + BEST_MODELS_COORDINATION_INDEX_SAMPLING_FOLDER + f"/{d_name}-{l}.png")