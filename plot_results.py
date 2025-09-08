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
    if not dict:
        return dict
    
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
num_samples = 64

# Datasets and tasks (we name them labels)
dataset_names = ["amsterdam", "nottingham"]
labels = ["pick", "guess", "coordinate"]

# Matplotlib config
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'

# Savedir (creation)
SAVEDIR = "./plots"
COORDINATION_INDEX_FOLDER = "/coordination-index/sample/llama"
BEST_MODELS_COORDINATION_INDEX_FOLDER = "/coordination-index-best/sample/llama"
BEST_MODELS_COORDINATION_INDEX_SAMPLING_FOLDER = "/coordination-index-best-merge/sample/llama"
BEST_MODELS_COORDINATION_INDEX_SAMPLING_FOLDER_SIDE = "/coordination-index-best-merge-side/sample/llama"


for folder in [COORDINATION_INDEX_FOLDER, 
               BEST_MODELS_COORDINATION_INDEX_FOLDER,
               BEST_MODELS_COORDINATION_INDEX_SAMPLING_FOLDER,
               BEST_MODELS_COORDINATION_INDEX_SAMPLING_FOLDER_SIDE]:
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
                
                data_llms[model_name] = []
                for i in range(0, len(data_llm), 3):
                    current_responses = {}
                    for d in data_llm[i:i+1]:
                        # For each task, put together the tasks and then average
                        for response in d["responses"].keys():
                            match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

                            if match:
                                re_response = match.group(1).strip().lower()
                                if re_response not in current_responses:
                                    current_responses[re_response] = 0
                                current_responses[re_response] += d["responses"][response]
                            else:
                                pass
                    
                    # Normalised coordination index
                    current_responses = sample(current_responses, n=num_samples)
                    N = sum(list(current_responses.values()))
                    n = len(current_responses.keys())
                    nci = 0.
                    if N > 1:
                        for _,v in current_responses.items():
                            nci += v*(v-1)
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
            plt.savefig(SAVEDIR + COORDINATION_INDEX_FOLDER + f"/{d_name}-{l}.png")
            plt.show()
            
    # 2. Coordination Index -- Best llamas
    model_names = [
        "/meta-llama/Meta-Llama-3-70B-Instruct",
        "/meta-llama/Llama-3.1-70B-Instruct",
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
                
                data_llms[model_name] = []
                for i in range(0, len(data_llm), 3):
                    current_responses = {}
                    for d in data_llm[i:i+1]:
                        # For each task, put together the tasks and then average
                        for response in d["responses"].keys():
                            match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

                            if match:
                                re_response = match.group(1).strip().lower()
                                if re_response not in current_responses:
                                    current_responses[re_response] = 0
                                current_responses[re_response] += d["responses"][response]
                            else:
                                pass
                    
                    # Normalised coordination index
                    current_responses = sample(current_responses, n=num_samples)
                    N = sum(list(current_responses.values()))
                    n = len(current_responses.keys())
                    nci = 0.
                    if N > 1:
                        for _,v in current_responses.items():
                            nci += v*(v-1)
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
            plt.savefig(SAVEDIR + BEST_MODELS_COORDINATION_INDEX_FOLDER + f"/{d_name}-{l}.png")
            plt.show()
            
    # 3. Coordination Index -- Merge best llamas
    model_names = [
        "/meta-llama/Meta-Llama-3-70B-Instruct",
        "/meta-llama/Llama-3.1-70B-Instruct",
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
                
                data_llms["meta-llama"] = []
                for i in range(0, len(data_llm), 3):
                    current_responses = {}
                    for d in data_llm[i:i+1]:
                        # For each task, put together the tasks and then average
                        for response in d["responses"].keys():
                            match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

                            if match:
                                re_response = match.group(1).strip().lower()
                                if re_response not in current_responses:
                                    current_responses[re_response] = 0
                                current_responses[re_response] += d["responses"][response]
                            else:
                                pass
                    
                    # Normalised coordination index
                    current_responses = sample(current_responses, n=num_samples)
                    N = sum(list(current_responses.values()))
                    n = len(current_responses.keys())
                    nci = 0.
                    if N > 1:
                        for _,v in current_responses.items():
                            nci += v*(v-1)
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

            tasks = [f"T{d_name[0].upper()}{i+1}" for i in range(14)]
            x = np.arange(len(tasks))  # label locations
            width = 0.3  # width of each bar

            # Plot human data (shifted left)
            human_bars = plt.bar(x - width/2, current_data_humans, width, label="Humans", color="black")

            # Plot meta-llama data (shifted right)
            current_data_llm = data_llms["meta-llama"]
            llm_bars = plt.bar(x + width/2, current_data_llm, width, label="meta-llama", color="blue", edgecolor="black")

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
            plt.savefig(SAVEDIR + BEST_MODELS_COORDINATION_INDEX_SAMPLING_FOLDER + f"/{d_name}-{l}.png")
            plt.show()


    # 4. Coordination Index -- Merge best llamas all tasks together
    # Amsterdam
    # Human data
    human_data = {'TA1': {'pick': 0.935, 'guess': 1.04, 'coordinate': 2.125},
                'TA2': {'pick': 1.04, 'guess': 1.124, 'coordinate': 1.472},
                'TA3': {'pick': 1.115, 'guess': 1.255, 'coordinate': 1.355},
                'TA4': {'pick': 1.064, 'guess': 1.112, 'coordinate': 0.984},
                'TA5': {'pick': 1.18, 'guess': 1.504, 'coordinate': 1.684},
                'TA6': {'pick': 0.985, 'guess': 1.02, 'coordinate': 2.2},
                'TA7': {'pick': 1.136, 'guess': 1.148, 'coordinate': 1.156},
                'TA8': {'pick': 1.21, 'guess': 1.7, 'coordinate': 2.495},
                'TA9': {'pick': 1.41, 'guess': 1.42, 'coordinate': 2.365},
                'TA10': {'pick': 1.06, 'guess': 1.168, 'coordinate': 2.208},
                'TA11': {'pick': 1.116, 'guess': 1.104, 'coordinate': 2.008},
                'TA12': {'pick': 1.235, 'guess': 2.095, 'coordinate': 2.36},
                'TA13': {'pick': 1.076, 'guess': 0.98, 'coordinate': 1.384},
                'TA14': {'pick': 1.104, 'guess': 1.148, 'coordinate': 2.072}}

    # LLM data
    llm_data = {'TA1': {'pick': 1.0, 'guess': 1.64, 'coordinate': 0.9931},
                'TA2': {'pick': 1.0, 'guess': 1.0, 'coordinate': 1.0},
                'TA3': {'pick': 1.6, 'guess': 1.38, 'coordinate': 1.0198},
                'TA4': {'pick': 1.0, 'guess': 1.13, 'coordinate': 1.0},
                'TA5': {'pick': 1.0, 'guess': 1.08, 'coordinate': 1.0},
                'TA6': {'pick': 1.27, 'guess': 1.82, 'coordinate': 1.24},
                'TA7': {'pick': 1.02, 'guess': 1.6, 'coordinate': 1.88},
                'TA8': {'pick': 1.06, 'guess': 2.12, 'coordinate': 0.9841},
                'TA9': {'pick': 1.0, 'guess': 1.32, 'coordinate': 1.05},
                'TA10': {'pick': 1.0, 'guess': 0.98, 'coordinate': 1.48},
                'TA11': {'pick': 1.31, 'guess': 1.31, 'coordinate': 1.0},
                'TA12': {'pick': 1.0, 'guess': 2.02, 'coordinate': 1.0},
                'TA13': {'pick': 1.0, 'guess': 2.09, 'coordinate': 1.0},
                'TA14': {'pick': 1.1, 'guess': 1.02, 'coordinate': 1.08}}

    TAs = list(human_data.keys())
    tasks = ['pick', 'guess', 'coordinate']

    x = np.arange(len(TAs))  # the label locations
    width = 0.35  # width of bar group

    fig, ax = plt.subplots(figsize=(14,6))

    # Human bars (black)
    human_vals = np.array([[human_data[ta][task] for task in tasks] for ta in TAs])
    ax.bar(x - width/2, human_vals[:,0], width/3, color='black',edgecolor='black', label='Human Pick')
    ax.bar(x - width/2 + width/3, human_vals[:,1], width/3, color='black', edgecolor='black', alpha=0.7, label='Human Guess')
    ax.bar(x - width/2 + 2*width/3, human_vals[:,2], width/3, color='black',edgecolor='black', alpha=0.4, label='Human Coordinate')

    # LLM bars (blue)
    llm_vals = np.array([[llm_data[ta][task] for task in tasks] for ta in TAs])
    ax.bar(x + width/2, llm_vals[:,0], width/3, color='blue',edgecolor='black', label='Llama Pick')
    ax.bar(x + width/2 + width/3, llm_vals[:,1], width/3, color='blue', edgecolor='black', alpha=0.7, label='Llama Guess')
    ax.bar(x + width/2 + 2*width/3, llm_vals[:,2], width/3, color='blue',edgecolor='black', alpha=0.4, label='Llama Coordinate')

    # Labels and formatting
    ax.set_ylabel('Normalised Coordination Index')
    ax.set_xlabel('TA')
    ax.set_title('Human vs LLM coordination indices for Amsterdam')
    ax.set_xticks(x + width/6)
    ax.set_xticklabels(TAs)
    ax.legend(ncol=2, fontsize=9)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(SAVEDIR + BEST_MODELS_COORDINATION_INDEX_SAMPLING_FOLDER_SIDE + f"/amsterdam.png")
    plt.show()
    

    # Nottingham
    # Human data for Nottingham
    human_data = {'TN1': {'pick': 1.235, 'guess': 3.22, 'coordinate': 3.03},
                'TN2': {'pick': 1.195, 'guess': 1.77, 'coordinate': 2.855},
                'TN3': {'pick': 1.43, 'guess': 2.36, 'coordinate': 2.15},
                'TN4': {'pick': 1.235, 'guess': 1.36, 'coordinate': 1.1425},
                'TN5': {'pick': 1.145, 'guess': 1.59, 'coordinate': 1.625},
                'TN6': {'pick': 1.17, 'guess': 1.695, 'coordinate': 1.355},
                'TN7': {'pick': 1.2, 'guess': 2.775, 'coordinate': 2.88},
                'TN8': {'pick': 1.07, 'guess': 2.315, 'coordinate': 1.92},
                'TN9': {'pick': 1.23, 'guess': 1.78, 'coordinate': 2.065},
                'TN10': {'pick': 0.93, 'guess': 0.98, 'coordinate': 1.105},
                'TN11': {'pick': 1.465, 'guess': 3.585, 'coordinate': 4.335},
                'TN12': {'pick': 1.17, 'guess': 1.38, 'coordinate': 1.61},
                'TN13': {'pick': 0.995, 'guess': 1.26, 'coordinate': 2.275},
                'TN14': {'pick': 1.385, 'guess': 1.675, 'coordinate': 2.13}}

    # LLM data for Nottingham
    llm_data = {'TN1': {'pick': 1.144841, 'guess': 1.07, 'coordinate': 1.49},
                'TN2': {'pick': 1.77, 'guess': 0.98, 'coordinate': 1.01},
                'TN3': {'pick': 1.555556, 'guess': 1.34, 'coordinate': 1.01},
                'TN4': {'pick': 2.31, 'guess': 1.0, 'coordinate': 1.0},
                'TN5': {'pick': 1.56, 'guess': 1.31, 'coordinate': 1.0},
                'TN6': {'pick': 1.0, 'guess': 1.0, 'coordinate': 1.0},
                'TN7': {'pick': 1.0, 'guess': 1.8, 'coordinate': 1.28},
                'TN8': {'pick': 1.51, 'guess': 1.15, 'coordinate': 1.0},
                'TN9': {'pick': 1.0, 'guess': 1.0, 'coordinate': 1.0},
                'TN10': {'pick': 1.0, 'guess': 1.0, 'coordinate': 1.0},
                'TN11': {'pick': 1.02, 'guess': 1.83, 'coordinate': 1.65},
                'TN12': {'pick': 1.0, 'guess': 1.0, 'coordinate': 1.0},
                'TN13': {'pick': 1.76, 'guess': 1.82, 'coordinate': 1.38},
                'TN14': {'pick': 1.0, 'guess': 1.0, 'coordinate': 1.0}}

    TAs = list(human_data.keys())
    tasks = ['pick', 'guess', 'coordinate']

    x = np.arange(len(TAs))  # the label locations
    width = 0.35  # width of bar group

    fig, ax = plt.subplots(figsize=(14,6))

    # Human bars (black)
    human_vals = np.array([[human_data[ta][task] for task in tasks] for ta in TAs])
    ax.bar(x - width/2, human_vals[:,0], width/3, color='black', edgecolor='black', label='Human Pick')
    ax.bar(x - width/2 + width/3, human_vals[:,1], width/3, color='black', edgecolor='black', alpha=0.7, label='Human Guess')
    ax.bar(x - width/2 + 2*width/3, human_vals[:,2], width/3, color='black', edgecolor='black', alpha=0.4, label='Human Coordinate')

    # LLM bars (blue)
    llm_vals = np.array([[llm_data[ta][task] for task in tasks] for ta in TAs])
    ax.bar(x + width/2, llm_vals[:,0], width/3, color='blue', edgecolor='black', label='Llama Pick')
    ax.bar(x + width/2 + width/3, llm_vals[:,1], width/3, color='blue', edgecolor='black', alpha=0.7, label='Llama Guess')
    ax.bar(x + width/2 + 2*width/3, llm_vals[:,2], width/3, color='blue', edgecolor='black', alpha=0.4, label='Llama Coordinate')

    # Labels and formatting
    ax.set_ylabel('Normalised Coordination Index')
    ax.set_xlabel('TN')
    ax.set_title('Human vs LLM coordination indices for Nottingham')
    ax.set_xticks(x + width/6)
    ax.set_xticklabels(TAs)
    ax.legend(ncol=2, fontsize=9)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(SAVEDIR + BEST_MODELS_COORDINATION_INDEX_SAMPLING_FOLDER_SIDE + f"/nottingham.png")
    plt.show()

