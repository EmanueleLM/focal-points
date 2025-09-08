import argparse
import json
import os
import matplotlib.pyplot as plt
import re
from collections import Counter

from src.utils import plot_block_frequencies

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", dest="model_name", type=str, default='meta-llama/Llama-3.2-1B-Instruct',
                    help="HuggingFace Model.")
parser.add_argument("-d", "--dataset", dest="dataset", type=str, default='schelling',
                    help="Jsonl dataset. Options are 'schelling'.")
parser.add_argument("-p", "--problem-tag", dest="problem_tag", type=str, default='problem',
                    help="Key of the problem in the json data. Options are 'problem', 'problem-nwp'.")

args = parser.parse_args()

os.makedirs(f"./images/{args.model_name}/{args.dataset}/{args.problem_tag}/", exist_ok=True)
os.makedirs(f"./results/{args.model_name}/", exist_ok=True)

with open(f"./logs/{args.model_name}/{args.dataset}_responses_{args.problem_tag}.jsonl", "r") as f:
    data = json.load(f)

plot_block_frequencies(data, args.dataset, args.model_name, args.problem_tag)