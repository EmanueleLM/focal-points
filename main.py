import argparse
import json
import os
import time
from src.llm import LLM
from src.prompt import *
from src.utils import iterate_data

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", dest="model_name", type=str, default='meta-llama/Llama-3.2-1B-Instruct',
                    help="HuggingFace Model.")
parser.add_argument("-t", "--trials", dest="trials", type=int, default=1,
                    help="Number of trials per experiment in dataset.")
parser.add_argument("-s", "--return-sequences", dest="sequences", type=int, default=30,
                    help="Number of responses per input.")
parser.add_argument("-d", "--dataset", dest="dataset", type=str, default='schelling',
                    help="Jsonl dataset. Options are 'schelling'.")
parser.add_argument("-q", "--quantization", dest="quantization", type=str, default=None,
                    help="Quantization. Options are 'None', '8bit', etc.")
parser.add_argument("-p", "--problem-tag", dest="problem_tag", type=str, default='problem',
                    help="Key of the problem in the json data. Options are 'problem', 'problem-nwp'.")

args = parser.parse_args()
model_name = args.model_name
trials = args.trials
dataset = args.dataset
quantization = args.quantization
problem_tag = args.problem_tag
num_return_sequences = args.sequences

# Global variables
dataset_dir = './data/'
logs_dir = f'./logs/{model_name}/'
for le_dir in [logs_dir, dataset_dir]:
    os.makedirs(le_dir, exist_ok=True)

# Load the model
model = LLM(
    model_id=model_name,
    num_return_sequences=num_return_sequences,
    quantization=quantization
)

# Load the dataset
with open(f"{dataset_dir}{dataset}.jsonl", 'r') as file:
    data = json.load(file)

problems, normalization_factors = iterate_data(data, problem_tag)

# Build a flat list of all (idx, problem) prompts
all_prompts = []
keys = []

for idx in problems:
    for problem in problems[idx]:
        full_prompt = Level0.prefix + problem + Level0.suffix
        all_prompts.append(full_prompt)
        keys.append((idx, problem))

all_outputs = []
for idx, problem in enumerate(all_prompts):
    outs = []
    for t in range(trials):
        print(f"Trial {t + 1}/{trials} for question number {idx + 1}/{len(all_prompts)},",
              f"generating {num_return_sequences} responses for question: \n{problem}\n")
        texts = model.generate_batch([problem])[0]
        outs = outs + texts
        print("\n".join(texts))
    all_outputs.append(outs)

# Reconstruct the original nested-dict format
responses = {}
for (idx, problem), prompt_outputs in zip(keys, all_outputs):
    if idx not in responses:
        responses[idx] = {}
    responses[idx][problem] = prompt_outputs

# Save the responses in a structured format  
logs = []
for idx in responses:
    for (variation_idx, problem), normalization_factor in zip(enumerate(responses[idx]), normalization_factors[idx]):
        log = {
            'idx': idx,
            'variation-idx': str(variation_idx),
            'prompt': problem,
            "responses": responses[idx][problem],
            'normalization_factor': normalization_factor
        }
        logs.append(log)

output_path = f'{logs_dir}{dataset}_responses_{problem_tag}.jsonl'
with open(output_path, 'w') as f:
    json.dump(logs, f, indent=2)

print(f"Responses saved to: {output_path}")

elapsed = int(time.time() - start_time)
print(f"Total time taken: {elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d} hours")
