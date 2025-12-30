#!/bin/bash
# GAMES=(
# "humans" 
# "p1_greedy"
# "p2_greedy"
# "both_greedy"  # may break the plots limits!
# "p1_cooperative"
# "p2_cooperative"
# "p1_SVO"
# "p2_SVO"
# "p1_adaptive"
# "p2_adaptive"
# "p1_llm"
# "p2_llm"
# )

# NUM_SAMPLES=100
# SAMPLE_WITH_REPLACEMENT=False
# for STRATEGY in "${GAMES[@]}"; do
#     echo "[bash] Running bargaining table analysis for strategy: $STRATEGY, num_samples: $NUM_SAMPLES, sample_with_replacement: $SAMPLE_WITH_REPLACEMENT"
#     python3 bargaining_table.py --strategy "$STRATEGY" --num-samples "$NUM_SAMPLES" --sample-with-replacement "$SAMPLE_WITH_REPLACEMENT"
# done

FILES=(
"./data/bargaining_table_llms/orange/gpt-oss-20b-low/bargaining_table_realdata_responses_vanilla_yellow.jsonl"
"./data/bargaining_table_llms/orange/gpt-oss-20b-low/bargaining_table_realdata_responses_saliency_yellow.jsonl"
"./data/bargaining_table_llms/orange/gpt-oss-20b-low/bargaining_table_realdata_responses_greedy_yellow.jsonl"
"./data/bargaining_table_llms/orange/gpt-oss-20b-low/bargaining_table_realdata_responses_cooperative_yellow.jsonl"
"./data/bargaining_table_llms/orange/gpt-oss-20b-low/bargaining_table_realdata_responses_all-features_yellow.jsonl"
"./data/bargaining_table_llms/orange/gpt-oss-20b-high/bargaining_table_realdata_responses_vanilla_yellow.jsonl"
"./data/bargaining_table_llms/orange/gpt-oss-20b-high/bargaining_table_realdata_responses_saliency_yellow.jsonl"
"./data/bargaining_table_llms/orange/gpt-oss-20b-high/bargaining_table_realdata_responses_greedy_yellow.jsonl"
"./data/bargaining_table_llms/orange/gpt-oss-20b-high/bargaining_table_realdata_responses_cooperative_yellow.jsonl"
"./data/bargaining_table_llms/orange/gpt-oss-20b-high/bargaining_table_realdata_responses_all-features_yellow.jsonl"
"./data/bargaining_table_llms/orange/gpt-oss-120b-low/bargaining_table_realdata_responses_vanilla_yellow.jsonl"
"./data/bargaining_table_llms/orange/gpt-oss-120b-low/bargaining_table_realdata_responses_saliency_yellow.jsonl"
"./data/bargaining_table_llms/orange/gpt-oss-120b-low/bargaining_table_realdata_responses_greedy_yellow.jsonl"
"./data/bargaining_table_llms/orange/gpt-oss-120b-low/bargaining_table_realdata_responses_cooperative_yellow.jsonl"
"./data/bargaining_table_llms/orange/gpt-oss-120b-low/bargaining_table_realdata_responses_all-features_yellow.jsonl"
"./data/bargaining_table_llms/orange/gpt-oss-120b-high/bargaining_table_realdata_responses_vanilla_yellow.jsonl"
"./data/bargaining_table_llms/orange/gpt-oss-120b-high/bargaining_table_realdata_responses_saliency_yellow.jsonl"
"./data/bargaining_table_llms/orange/gpt-oss-120b-high/bargaining_table_realdata_responses_greedy_yellow.jsonl"
"./data/bargaining_table_llms/orange/gpt-oss-120b-high/bargaining_table_realdata_responses_cooperative_yellow.jsonl"
"./data/bargaining_table_llms/orange/gpt-oss-120b-high/bargaining_table_realdata_responses_all-features_yellow.jsonl"
)

STRATEGY="p2_llm"

NUM_SAMPLES=100
SAMPLE_WITH_REPLACEMENT=False
for FILE in "${FILES[@]}"; do
    echo "[bash] Running bargaining table analysis for strategy: $STRATEGY, num_samples: $NUM_SAMPLES, sample_with_replacement: $SAMPLE_WITH_REPLACEMENT, --file-player-as-llm $FILE"
    python3 bargaining_table.py --strategy "$STRATEGY" --num-samples "$NUM_SAMPLES" --sample-with-replacement "$SAMPLE_WITH_REPLACEMENT" --file-player-as-llm "$FILE"
done
