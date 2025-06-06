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

args = parser.parse_args()
model_name = args.model_name
dataset = args.dataset

os.makedirs(f"./images/{model_name}/{dataset}/", exist_ok=True)
os.makedirs(f"./results/{model_name}/", exist_ok=True)

with open(f"./logs/{model_name}/{dataset}_responses.jsonl", "r") as f:
    data = json.load(f)
    
plot_block_frequencies(data, dataset, model_name)