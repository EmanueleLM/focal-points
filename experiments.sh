#!/bin/bash

# Default arguments
models=("meta-llama/Llama-3.3-70B-Instruct")
num_experiments=100
quantization="None"

# Parse command-line arguments
while getopts "m:n:q:" opt; do
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
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Handle "all-models" option
if [[ "$models" == "all-models" ]]; then
  models=("meta-llama/Llama-3.2-3B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "microsoft/Phi-4-mini-instruct" "google/gemma-3-4b-it")
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
  #     python main.py --model "$model" --dataset "$data" --problem-tag "$ptag" --return-sequences "$num_experiments" --quantization "$quantization"

  #     echo "Running metrics.py with $data and $ptag"
  #     python metrics.py --model "$model" --dataset "$data" --problem-tag "$ptag"
  #   done
  # done

  # # le Amsterdam and Nottingham
  # datasets=("amsterdam" "nottingham")
  # problemtags=("problem-pick" "problem-guess" "problem-coordinate")

  # for data in "${datasets[@]}"; do
  #   for ptag in "${problemtags[@]}"; do
  #     echo "Running main.py with $data and $ptag"
  #     python main.py --model "$model" --dataset "$data" --problem-tag "$ptag" --return-sequences "$num_experiments" --quantization "$quantization"

  #     echo "Running metrics.py with $data and $ptag"
  #     python metrics.py --model "$model" --dataset "$data" --problem-tag "$ptag"
  #   done
  # done

  # le Amsterdam and Nottingham numeric
  datasets=("amsterdam_numeric" "nottingham_numeric")
  problemtags=("problem-pick" "problem-guess" "problem-coordinate")

  for data in "${datasets[@]}"; do
    for ptag in "${problemtags[@]}"; do
      echo "Running main.py with $data and $ptag"
      python main.py --model "$model" --dataset "$data" --problem-tag "$ptag" --return-sequences "$num_experiments" --quantization "$quantization"

      echo "Running metrics.py with $data and $ptag"
      python metrics.py --model "$model" --dataset "$data" --problem-tag "$ptag"
    done
  done

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