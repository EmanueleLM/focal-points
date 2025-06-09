#!/bin/bash
models=("meta-llama/Llama-3.2-3B-Instruct")
num_experiments=100

for model in "${models[@]}"; do

  echo "Running experiments with model: $model"

  # le Schelling
  datasets=("schelling")
  problemtags=("problem-nwp" "problem")

  for data in "${datasets[@]}"; do
    for ptag in "${problemtags[@]}"; do
      echo "Running main.py with $data and $ptag"
      python main.py --model "$model" --dataset "$data" --problem-tag "$ptag" --return-sequences "$num_experiments"

      echo "Running metrics.py with $data and $ptag"
      python metrics.py --model "$model" --dataset "$data" --problem-tag "$ptag"
    done
  done

  # le Amsterdam and Nottingham
  datasets=("amsterdam" "nottingham")
  problemtags=("problem-pick" "problem-guess" "problem-coordinate")

  for data in "${datasets[@]}"; do
    for ptag in "${problemtags[@]}"; do
      echo "Running main.py with $data and $ptag"
      python main.py --model "$model" --dataset "$data" --problem-tag "$ptag" --return-sequences "$num_experiments"

      echo "Running metrics.py with $data and $ptag"
      python metrics.py --model "$model" --dataset "$data" --problem-tag "$ptag"
    done
  done
done
