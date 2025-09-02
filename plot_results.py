import glob
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from collections import defaultdict
from pathlib import Path

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

# Matplotlib config
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'

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
models = ["meta-llama"
          "google", 
          "microsoft", 
          "Qwen"
          ]

# Datasets and tasks (we name them labels)
dataset_names = ["amsterdam", "nottingham"]
labels = ["pick", "guess", "coordinate"]

# Savedir (creation)
SAVEDIR = "./plots"
COORDINATION_INDEX_FOLDER = "/coordination-index"
FAMILY_COORDINATION_INDEX_FOLDER = "/family-coordination-index"
BEST_MODELS_COORDINATION_INDEX_FOLDER = "/best-models-coordination-index"

for folder in [COORDINATION_INDEX_FOLDER, 
               FAMILY_COORDINATION_INDEX_FOLDER, 
               BEST_MODELS_COORDINATION_INDEX_FOLDER]:
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
    
    # 1. Coordination Index -- Each LLM has its own line
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
                            if response not in current_responses:
                                current_responses[response] = 0
                            current_responses[response] += d["responses"][response]
                    # Normalised coordination index
                    nci = 0.
                    N = sum(list(current_responses.values()))
                    n = len(current_responses)
                    for _,v in current_responses.items():
                        nci += v*(v-1)
                    print(N, n)
                    data_llms[model_name].append(nci*n/(N*(N-1)))

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
            
    # 2. Average Coordination Index -- Each family has its own line
    # Each family's coordination index is the average of its members
    for d_name in dataset_names:
        print(f"Dataset: {d_name}")
        for l in labels:
            print(f"Label: {l}")

            # Load LLM results
            data_llms = {}
            for model_name in model_names:
                with open(f"./results/{model_name}/{d_name}_problem-{l}.jsonl", "r") as f:
                    data_llm = json.load(f)
                data_llms[model_name] = [
                    sum([d["normalised_coordination_index"] for d in data_llm[i:i+3]])/3
                    for i in range(0, len(data_llm), 3)
                ]

            # Load Human results
            with open(f"./data/Bardsley-humans/{d_name}.jsonl", "r") as f:
                data_humans = json.load(f)
            current_data_humans = [
                d["normalised_coordination_index"] for d in data_humans if d["task"] == l
            ]

            # --- Plot ---
            plt.figure(figsize=(10, 5))
            plt.plot(current_data_humans, label="Humans", marker="o", color="black")

            # Overall LLM average
            all_data = [data_llms[m] for m in model_names]
            plt.plot(np.mean(all_data, axis=0), label="LLMs - Average", marker="x", color="gray")

            # Family averages
            family_data = defaultdict(list)
            for m in model_names:
                family = get_family(m)
                family_data[family].append(data_llms[m])

            for family, curves in family_data.items():
                avg_curve = np.mean(curves, axis=0)
                cmap = family_colormaps[family]
                plt.plot(avg_curve, label=f"{family} (avg)", marker="s", color=cmap(0.7))

            plt.xticks([i for i in range(14)], [f"T{d_name[0].upper()}{i+1}" for i in range(14)])
            plt.title(f"Coordination Index Comparison: {d_name.capitalize()}-{l}")
            plt.ylabel("Normalised Coordination Index")
            plt.legend(bbox_to_anchor=(1, 0.5))
            plt.grid()
            plt.savefig(SAVEDIR + FAMILY_COORDINATION_INDEX_FOLDER + f"/{d_name}-{l}.png")
            
    # 3. Coordination Index Between best models
    # In this case, we do not average but first merge and then compute the coordination index
    best_models = [
        "/meta-llama/Llama-3.3-70B-Instruct",
        "/google/gemma-3-4b-it",
        "/microsoft/Phi-4-mini-instruct",
        "/Qwen/Qwen2.5-14B-Instruct-1M"
    ]

    # Collect the normalisation factor for each task
    normalisation_factor = {}
    for d_name in dataset_names:
        normalisation_factor[d_name] = {}
        with open(f"./data/{d_name}.jsonl", "r") as f:
            data = json.load(f)
        
        for problem in data:
            normalisation_factor[d_name][problem["id"]] = int(problem["normalization_factor"])

    # Sample one model per family
    model_name_per_family = {}
    for m in best_models:
        # Collect prefixes
        family_name = m.split('/')[1]
        if family_name not in model_name_per_family:
            model_name_per_family[family_name] = []
        model_name_per_family[family_name].append(m)

    best_models_results = {}
    for d_name in dataset_names:
        print(f"Dataset: {d_name}")
        best_models_results[d_name] = {}
        for l in labels:
            print(f"Label: {l}")
            best_models_results[d_name][l] = {}
            # Load LLM results
            data_llms = {}
            for model_name in best_models:
                with open(f"./results/{model_name}/{d_name}_problem-{l}.jsonl", "r") as f:
                    data_llm = json.load(f)
                
                for i in range(0, len(data_llm), 3):
                    for d in data_llm[i:i+3]:
                        if d["idx"] not in best_models_results[d_name][l]:
                            best_models_results[d_name][l][d["idx"]] = {}
                        for response in d["responses"].keys():
                            if response not in best_models_results[d_name][l][d["idx"]].keys():
                                best_models_results[d_name][l][d["idx"]][response] = 0
                            best_models_results[d_name][l][d["idx"]][response] += int(d["responses"][response])

    # Compute the normalised coordination index
    normalised_coordination_index = {}                       
    for d_name in dataset_names:
        normalised_coordination_index[d_name] = {}
        for l in labels:
            llm_all_results = []
            normalised_coordination_index[d_name][l] = {}
            _ta_order = []
            for ta in best_models_results[d_name][l]:
                _ta_order.append(ta)
                N = sum(list(best_models_results[d_name][l][ta].values()))
                normalised_coordination_index[d_name][l][ta] = 0.
                for choice in best_models_results[d_name][l][ta]:
                    m_j = best_models_results[d_name][l][ta][choice]
                    normalised_coordination_index[d_name][l][ta] += (m_j * (m_j - 1))/(N*(N-1))
                    
                normalised_coordination_index[d_name][l][ta] *= normalisation_factor[d_name][ta]
                llm_all_results.append(normalised_coordination_index[d_name][l][ta])

            # Load Human results
            with open(f"./data/Bardsley-humans/{d_name}.jsonl", "r") as f:
                data_humans = json.load(f)
            current_data_humans = [
                d["normalised_coordination_index"] for d in data_humans if d["task"] == l
            ]

            # --- Plot ---
            plt.figure(figsize=(10, 5))
            plt.plot(current_data_humans, label="Humans", marker="o", color="black")

            # Best LLMs
            print(_ta_order)  # Should be in the correct order
            print(llm_all_results)
            plt.plot(llm_all_results, label="Best LLMs", marker="x", color="gray")

            plt.xticks([i for i in range(14)], [f"T{d_name[0].upper()}{i+1}" for i in range(14)])
            plt.title(f"Coordination Index Comparison: {d_name.capitalize()}-{l}")
            plt.ylabel("Inter-Models Normalised Coordination Index")
            plt.legend(bbox_to_anchor=(1, 0.5))
            plt.grid()
            plt.savefig(SAVEDIR + BEST_MODELS_COORDINATION_INDEX_FOLDER + f"/{d_name}-{l}.png")
    