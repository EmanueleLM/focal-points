import json
import matplotlib.pyplot as plt
import os

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'

models = ["meta-llama", "google", "microsoft", "Qwen"]
model_names = []

# Collect the names of the folders inside each "model"
for m in models:
    for model_path in next(os.walk("./results/" + m)):
        if "./results/" in model_path:
            prefix = model_path
            continue
        model_names.extend([prefix.split("./results")[1] + "/" + llm for llm in model_path])
        
dataset_names = ["amsterdam", "nottingham"]
labels = ["pick", "guess", "coordinate"]

print("--- Available Models ---")
for i,m in enumerate(model_names):
    print(f"{i+1}. {m}")
    
# Check which models we can actually use for the experiments
import glob

required_files = [
    "amsterdam_numeric_problem-coordinate.jsonl",
    "amsterdam_numeric_problem-guess.jsonl",
    "amsterdam_numeric_problem-pick.jsonl",
    "amsterdam_problem-coordinate.jsonl",
    "amsterdam_problem-guess.jsonl",
    "amsterdam_problem-pick.jsonl",
    "asymmetric_payoff_problem-coordinate.jsonl",
    "asymmetric_payoff_problem-guess.jsonl",
    "asymmetric_payoff_problem-pick.jsonl",
    "nottingham_numeric_problem-coordinate.jsonl",
    "nottingham_numeric_problem-guess.jsonl",
    "nottingham_numeric_problem-pick.jsonl",
    "nottingham_problem-coordinate.jsonl",
    "nottingham_problem-guess.jsonl",
    "nottingham_problem-pick.jsonl",
    "schelling_problem.jsonl",
    "schelling_problem-nwp.jsonl"
]


# Scan each folder in model_names and check which files are missing:
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
        
model_names = working_model_names

best_models = [
    "/meta-llama/Llama-3.2-3B-Instruct",
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
        plt.savefig(f"./tmp/{d_name}{l}.png")
    