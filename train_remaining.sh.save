#!/bin/bash
# File: train_remaining.sh
# Purpose: Automated Sequential Runs for the final 9 models (Mac Local)

PARAMS=("buoyancy_offset" "added_mass" "act_efficiency")
SEEDS=(0 1 2)
STEPS=500000

# MAC LOCAL PATH
BASE_DIR="$HOME/rl_research/auv"

# Ensure the save directory exists before starting
mkdir -p "$BASE_DIR"

echo "=========================================================="
echo "STARTING FINAL 9 ABLATION RUNS (MAC LOCAL)"
echo "Saving models to: $BASE_DIR"
echo "Estimated time: ~6.5 Hours"
echo "=========================================================="

for PARAM in "${PARAMS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        
        # Remove underscores for a cleaner folder name
        CLEAN_PARAM=${PARAM//_/}
        RUN_NAME="ablation_${CLEAN_PARAM}_${SEED}"
        
        echo "----------------------------------------------------------"
        echo ">>> NOW TRAINING: $PARAM | SEED: $SEED <<<"
        echo "----------------------------------------------------------"
        
        python scripts/train.py \
            --mode uniform \
            --seed $SEED \
            --steps $STEPS \
            --ablation-param $PARAM \
            --run-name $RUN_NAME \
            --save-dir "$BASE_DIR"
            
    done
done

echo "=========================================================="
echo "✅ ALL 15 ABLATION RUNS COMPLETE! YOU CAN WAKE UP NOW."
echo "=========================================================="