#!/bin/bash
GAMES=(
"humans" 
"p1_greedy"
"p2_greedy"
"both_greedy"  # may break the plots limits!
"p1_cooperative"
"p2_cooperative"
"p1_SVO"
"p2_SVO"
"p1_adaptive"
"p2_adaptive"
"p1_llm"
"p2_llm"
)

NUM_SAMPLES=100
SAMPLE_WITH_REPLACEMENT=False
for STRATEGY in "${GAMES[@]}"; do
    echo "[bash] Running bargaining table analysis for strategy: $STRATEGY, num_samples: $NUM_SAMPLES, sample_with_replacement: $SAMPLE_WITH_REPLACEMENT"
    python3 bargaining_table.py --strategy "$STRATEGY" --num-samples "$NUM_SAMPLES" --sample-with-replacement "$SAMPLE_WITH_REPLACEMENT"
done