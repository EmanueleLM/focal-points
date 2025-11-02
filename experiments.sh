#!/bin/bash

# Default arguments
models=("meta-llama/Llama-3.3-70B-Instruct")
num_experiments=30
quantization="None"
plot_graphs="true"

# Parse command-line arguments
while getopts "m:n:q:p:" opt; do
  case $opt in
    m)
      models=("$OPTARG")
      ;;
    n)
      num_experiments=$OPTARG
      ;;
    q)
      quantization=$OPTARG
      ;;
    p)
      plot_graphs=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Handle "all-models" option
if [[ "$models" == "all" ]]; then
  models=(
  # "meta-llama/Llama-3.3-70B-Instruct"
  # "meta-llama/Llama-3.1-70B-Instruct"
  # "meta-llama/Meta-Llama-3-70B-Instruct"
  # "meta-llama/Llama-3.1-8B-Instruct"
  # "meta-llama/Meta-Llama-3-8B-Instruct"
  # "meta-llama/Llama-3.2-3B-Instruct"
  # "meta-llama/Llama-3.2-1B-Instruct"
  "Qwen/Qwen2-72B-Instruct"
  "Qwen/Qwen2.5-72B-Instruct"
  # "Qwen/Qwen2.5-32B-Instruct"
  # "Qwen/Qwen2.5-14B-Instruct-1M"
  # "Qwen/Qwen2.5-14B-Instruct"
  # "Qwen/Qwen2.5-7B-Instruct-1M"
  # "Qwen/Qwen2.5-7B-Instruct"
  # "Qwen/Qwen2-7B-Instruct"
  # "Qwen/Qwen2.5-3B-Instruct"
  # "Qwen/Qwen2.5-1.5B-Instruct"
  # "Qwen/Qwen2-1.5B-Instruct"
  # "Qwen/Qwen2.5-0.5B-Instruct"
  # "Qwen/Qwen2-0.5B-Instruct"
  # "google/gemma-3-1b-it"
  # "google/gemma-3-4b-it"
  # "google/gemma-3-12b-it"
  # "google/gemma-3-27b-it"
  # "microsoft/Phi-4-mini-instruct"
  "openai/gpt-oss-120b"
  )
else
  models=("$models")
fi

datasets=(
    # # TASK -- Amsterdam
    # "amsterdam"
    # "amsterdam-instruct-all-features"
    # "amsterdam-instruct-saliency"
    # # TASK -- Amsterdam_numeric
    # "amsterdam_numeric"
    # "amsterdam_numeric-instruct-all-features"
    # "amsterdam_numeric-instruct-saliency"
    # # TASK -- Asymmetric_payoff
    # "asymmetric_payoff"
    # "asymmetric_payoff-instruct-all-features"
    # "asymmetric_payoff-instruct-saliency"
    # # TASK -- Nottingham
    # "nottingham"
    # "nottingham-instruct-all-features"
    # "nottingham-instruct-saliency"
    # # TASK -- Nottingham_numeric
    # "nottingham_numeric"
    # "nottingham_numeric-instruct-all-features"
    # "nottingham_numeric-instruct-saliency"
    # TASK -- Amsterdam culture
    "amsterdam-instruct-culture"
    # TASK -- Nottingham culture
    "nottingham-instruct-culture"
  )

problemtags=("problem-pick" "problem-guess" "problem-coordinate")

echo "Running experiments with model: $models, quantization: $quantization, number of experiments: $num_experiments"

for model in "${models[@]}"; do
  echo "Using model: $model"

  for data in "${datasets[@]}"; do
    for ptag in "${problemtags[@]}"; do
      echo "Running main.py with $data and $ptag"
      python main.py --model "$model" --dataset "$data" --problem-tag "$ptag"  --return-sequences "$num_experiments" --quantization "$quantization" --plot-graphs "$plot_graphs"

    done
  done
done

# datasets=(
#     # schelling
#     "schelling"
#     "schelling-instruct-all-features"
#     "schelling-instruct-saliency"
#   )

# problemtags=("problem")

# for model in "${models[@]}"; do
#   echo "Using model: $model"

#   for data in "${datasets[@]}"; do
#     for ptag in "${problemtags[@]}"; do
#       echo "Running main.py with $data and $ptag"
#       python main.py --model "$model" --dataset "$data" --problem-tag "$ptag"  --return-sequences "$num_experiments" --quantization "$quantization" --plot-graphs "$plot_graphs"

#     done
#   done
# done