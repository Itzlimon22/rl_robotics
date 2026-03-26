# RL Robotics Research — Hybrid Setup

## Structure
```
rl_robotics/
├── scripts/
│   ├── train.py        # Main training — auto-detects MPS / CUDA / CPU
│   ├── eval.py         # Load & evaluate saved models
│   └── sync.sh         # Push code to GitHub for Colab pickup
├── configs/
│   ├── halfcheetah_sac.yaml
│   └── ant_sac.yaml
├── notebooks/
│   └── colab_trainer.py   # Colab cell templates
└── .gitignore
```

## Google Drive layout (auto-created)
```
MyDrive/rl_research/
└── HalfCheetah-v4/
    └── run_01/
        ├── best_model/       # Best checkpoint (eval)
        ├── checkpoints/      # Periodic saves (every 50k steps)
        ├── tb_logs/          # TensorBoard logs
        ├── eval_logs/        # Evaluation results
        ├── sac_final.zip     # Final trained model
        └── vec_normalize.pkl # Obs normalizer stats
```

## Workflow

### On Mac (develop & debug)
```bash
conda activate rl
python scripts/train.py --env HalfCheetah-v4 --total-steps 50000  # quick test
./scripts/sync.sh  # push to GitHub when ready for full run
```

### On Colab (full training)
1. Open Colab, mount Drive
2. `!git clone https://github.com/YOUR_USERNAME/rl_robotics`
3. Follow cells in notebooks/colab_trainer.py
4. Models auto-save to Drive — access from Mac anytime

### Load a model trained on Colab, eval on Mac
```bash
python scripts/eval.py --env HalfCheetah-v4 --run-name run_01
```
