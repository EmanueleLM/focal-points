import argparse
import json
import os

from src.llm import LLM
from src.prompt import *
from src.utils import iterate_schelling_data, coordination_index

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", dest="model_name", type=str, default='meta-llama/Llama-3.2-1B-Instruct',
                    help="HuggingFace Model.")
parser.add_argument("-t", "--trials", dest="trials", type=int, default=3,
                    help="Number of trials per experiment in dataset.")
parser.add_argument("-d", "--dataset", dest="dataset", type=str, default='schelling',
                    help="Jsonl dataset. Options are 'schelling'.")
parser.add_argument("-q", "--quantization", dest="quantization", type=str, default='None',
                    help="Quantization. Options are 'None', '8bit', etc.")
parser.add_argument("-p", "--problem-tag", dest="problem_tag", type=str, default='problem',
                    help="Key of the problem in the json data. Options are 'problem', 'problem-nwp'.")

args = parser.parse_args()
model_name = str(args.model_name)
trials = int(args.trials)
dataset = str(args.dataset)
quantization = str(args.quantization)
problem_tag = str(args.problem)

# Global variables
logs_dir = './logs/'
dataset_dir = './datasets/'
results_dir = './results/'

# Create the directories if they do not exist
for dir in [logs_dir, dataset_dir, results_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Load the model
model = LLM(
    model_id=model_name,
    num_return_sequences=trials,
    quantization=quantization
)

# Load the dataset
with open(f"{dataset_dir}{dataset}", 'r') as file:
    data = json.load(file)

problems = iterate_schelling_data(data, problem_tag)

# Build a flat list of all (idx, problem) prompts
batch_prompts = []
keys = []

for t in range(trials):
    print(f"Batch {t+1}/{trials}")
    for idx in problems:
        for problem in problems[idx]:
            full_prompt = Level0.prefix + problem + Level0.suffix
            batch_prompts.append(full_prompt)
            keys.append((idx, problem))

    # Send all prompts in one generate_batch call
    batch_outputs = model.generate_batch(batch_prompts)

# Reconstruct the original nested-dict format
responses = {}
for (idx, problem), prompt_outputs in zip(keys, batch_outputs):
    if idx not in responses:
        responses[idx] = {}
    responses[idx][problem] = prompt_outputs

# Save the responses in a structured format  
logs = []
for idx in responses:
    for problem in responses[idx]:
        log = {
            'idx': idx,
            'prompt': problem,
            "responses": responses[idx][problem]
        }

        logs.append(log)

with open(f'{logs_dir}{dataset}_responses.jsonl', 'w') as f:
    json.dump(logs, f, indent=2)