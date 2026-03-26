#!/bin/bash
# sync.sh — Push local code to GitHub so Colab can pull it
# Usage: ./scripts/sync.sh "optional commit message"

MSG=${1:-"sync: $(date '+%Y-%m-%d %H:%M')"}
echo "🔄  Committing: $MSG"
git add -A
git commit -m "$MSG"
git push origin main
echo "✅  Done — on Colab run: !git clone https://github.com/YOUR_USERNAME/rl_robotics"
