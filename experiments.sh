#!/bin/bash

# le Schelling
datasets=("schelling")
problemtags=("problem-nwp" "problem")

for data in "${datasets[@]}"; do
  for ptag in "${problemtags[@]}"; do
    echo "Running main.py with $data and $ptag"
    python main.py --dataset "$data" --problem-tag "$ptag" --return-sequences 30

    echo "Running metrics.py with $data and $ptag"
    python metrics.py --dataset "$data" --problem-tag "$ptag"
  done
done

# le Amsterdam and Nottingham
datasets=("amsterdam" "nottingham")
problemtags=("problem-pick" "problem-guess" "problem-coordinate")

for data in "${datasets[@]}"; do
  for ptag in "${problemtags[@]}"; do
    echo "Running main.py with $data and $ptag"
    python main.py --dataset "$data" --problem-tag "$ptag" --return-sequences 30

    echo "Running metrics.py with $data and $ptag"
    python metrics.py --dataset "$data" --problem-tag "$ptag"
  done
done
