#!/bin/bash
# sync_vault.sh - Automates the backup of all critical paper assets.

echo "🗄️ Initializing Paper Vault..."

VAULT_DIR="$HOME/rl_research/paper_assets"
BASE_DIR="$HOME/rl_research/auv"

# 1. Create directory structure
mkdir -p "$VAULT_DIR/models"
mkdir -p "$VAULT_DIR/videos"

# 2. Extract best models and vecnorms safely
echo "💾 Backing up Champion Models..."
for mode in none uniform curriculum; do
    for seed in 0 1 2 3; do
        RUN_DIR="$BASE_DIR/master_${mode}/master_${mode}_seed${seed}"
        if [ -d "$RUN_DIR" ]; then
            # Create a specific folder in the vault for this model
            TARGET_DIR="$VAULT_DIR/models/master_${mode}_seed${seed}"
            mkdir -p "$TARGET_DIR"
            
            cp "$RUN_DIR/best_model.zip" "$TARGET_DIR/" 2>/dev/null
            cp "$RUN_DIR/vec_normalize.pkl" "$TARGET_DIR/" 2>/dev/null
        fi
    done
done

# 3. Extract all rendered MP4 videos
echo "🎥 Gathering Rendered Videos..."
find "$BASE_DIR" -type f -name "*.mp4" -exec cp {} "$VAULT_DIR/videos/" \;

echo "✅ Vault Synchronization Complete! Your data is safe."