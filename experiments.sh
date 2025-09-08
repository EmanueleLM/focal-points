#!/bin/bash

# Default arguments
models=("meta-llama/Llama-3.3-70B-Instruct")
num_experiments=5
quantization="None"
plot_graphs=true

# Parse command-line arguments
while getopts "m:n:q:pg" opt; do
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
    pg)
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
  "meta-llama/Llama-3.3-70B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.1-70B-Instruct"
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "meta-llama/Meta-Llama-3-70B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct-1M"
  "Qwen/Qwen2.5-14B-Instruct-1M"
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
  "Qwen/Qwen2.5-72B-Instruct"
  "Qwen/Qwen2-72B-Instruct"
  "Qwen/Qwen2-7B-Instruct"
  "Qwen/Qwen2-1.5B-Instruct"
  "Qwen/Qwen2-0.5B-Instruct"
  "google/gemma-3-1b-it"
  "google/gemma-3-4b-it"
  "google/gemma-3-12b-it"
  "google/gemma-3-27b-it"
  "microsoft/Phi-4-mini-instruct"
  )
else
  models=("$models")
fi

echo "Running experiments with model: $models, quantization: $quantization, number of experiments: $num_experiments"

for model in "${models[@]}"; do
  echo "Using model: $model"

  # # le Schelling
  # datasets=("schelling")
  # problemtags=("problem-nwp" "problem")

  # for data in "${datasets[@]}"; do
  #   for ptag in "${problemtags[@]}"; do
  #     echo "Running main.py with $data and $ptag"
  #     python main.py --model "$model" --dataset "$data" --problem-tag "$ptag"  --return-sequences "$num_experiments" --quantization "$quantization"

  #     echo "Running metrics.py with $data and $ptag"
  #     python metrics.py --model "$model" --dataset "$data" --problem-tag "$ptag"

  #   done
  # done

  # le Amsterdam and Nottingham
  datasets=("amsterdam" "nottingham")
  problemtags=("problem-pick" "problem-guess" "problem-coordinate")

  for data in "${datasets[@]}"; do
    for ptag in "${problemtags[@]}"; do
      echo "Running main.py with $data and $ptag"
      python main.py --model "$model" --dataset "$data" --problem-tag "$ptag" --return-sequences "$num_experiments" --quantization "$quantization"

      # echo "Running metrics.py with $data and $ptag"
      # python metrics.py --model "$model" --dataset "$data" --problem-tag "$ptag"

    done
  done

  # # le Amsterdam and Nottingham numeric
  # datasets=("amsterdam_numeric" "nottingham_numeric")
  # problemtags=("problem-pick" "problem-guess" "problem-coordinate")

  # for data in "${datasets[@]}"; do
  #   for ptag in "${problemtags[@]}"; do
  #     echo "Running main.py with $data and $ptag"
  #     python main.py --model "$model" --dataset "$data" --problem-tag "$ptag"  --return-sequences "$num_experiments" --quantization "$quantization"

  #     # echo "Running metrics.py with $data and $ptag"
  #     # python metrics.py --model "$model" --dataset "$data" --problem-tag "$ptag"

  #   done
  # done

  # # le Asymmetric Payoff
  # datasets=("asymmetric_payoff")
  # problemtags=("problem-pick" "problem-guess" "problem-coordinate")

  # for data in "${datasets[@]}"; do
  #   for ptag in "${problemtags[@]}"; do
  #     echo "Running main.py with $data and $ptag"
  #     python main.py --model "$model" --dataset "$data" --problem-tag "$ptag" --return-sequences "$num_experiments" --quantization "$quantization"

  #     echo "Running metrics.py with $data and $ptag"
  #     python metrics.py --model "$model" --dataset "$data" --problem-tag "$ptag"

  #   done
  # done
done