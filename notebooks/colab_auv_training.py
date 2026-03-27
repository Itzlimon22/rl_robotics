# ════════════════════════════════════════════════════════════════════════════
# Halcyon AUV — Colab Training Notebook
# ════════════════════════════════════════════════════════════════════════════
# Usage: Copy each cell block into a Colab cell and run top-to-bottom.
# Runtime: GPU (A100 recommended — Runtime > Change runtime type > A100)
# Estimated time: ~45 min per 1M-step run on A100
# ════════════════════════════════════════════════════════════════════════════


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELL 1 — Mount Google Drive                                             │
# └─────────────────────────────────────────────────────────────────────────┘

from google.colab import drive
drive.mount("/content/drive")

import os
DRIVE_ROOT = "/content/drive/MyDrive/rl_research/auv"
os.makedirs(DRIVE_ROOT, exist_ok=True)
print(f"Drive mounted. Save root: {DRIVE_ROOT}")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELL 2 — Install dependencies                                           │
# │ Run once per session. ~2 minutes.                                       │
# └─────────────────────────────────────────────────────────────────────────┘

# %%capture install_output
import subprocess, sys

packages = [
    "mujoco",
    "gymnasium[mujoco]",
    "stable-baselines3[extra]",
    "torch",
    "tensorboard",
]

for pkg in packages:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", pkg],
        check=True
    )

# Verify
import mujoco, gymnasium, stable_baselines3, torch
print(f"mujoco:           {mujoco.__version__}")
print(f"gymnasium:        {gymnasium.__version__}")
print(f"stable-baselines3:{stable_baselines3.__version__}")
print(f"torch:            {torch.__version__}")
print(f"CUDA available:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:              {torch.cuda.get_device_name(0)}")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELL 3 — Clone / sync your GitHub repo                                  │
# │ Option A: clone fresh (first time)                                      │
# │ Option B: pull latest (subsequent runs)                                 │
# └─────────────────────────────────────────────────────────────────────────┘

import os
from pathlib import Path

REPO_URL  = "https://github.com/Itzlimon22/Obhyash-complete-project"  # update if moved
REPO_DIR  = Path("/content/rl_robotics")

# ── Option A: clone fresh ────────────────────────────────────────────────
if not REPO_DIR.exists():
    os.system(f"git clone {REPO_URL} {REPO_DIR}")
    print(f"Cloned → {REPO_DIR}")
else:
    # ── Option B: pull latest ─────────────────────────────────────────────
    os.system(f"cd {REPO_DIR} && git pull")
    print(f"Pulled latest → {REPO_DIR}")

# Verify key files
for f in ["envs/auv.xml", "envs/auv_env.py", "envs/auv_dr_wrapper.py", "scripts/train.py"]:
    path = REPO_DIR / f
    status = "✓" if path.exists() else "✗ MISSING"
    print(f"  {status}  {f}")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELL 4 — Sanity check (run env for 100 steps, confirm everything works) │
# │ Do this before every long run. Costs ~10 seconds.                       │
# └─────────────────────────────────────────────────────────────────────────┘

import sys
sys.path.insert(0, str(REPO_DIR / "envs"))

from auv_env        import HalcyonAUVEnv
from auv_dr_wrapper import AUVDomainRandomWrapper

xml_path = str(REPO_DIR / "envs" / "auv.xml")
env = AUVDomainRandomWrapper(
    HalcyonAUVEnv(xml_path=xml_path),
    mode="curriculum",
    seed=0,
    verbose=False,
)

obs, info = env.reset(seed=0)
print(f"obs shape: {obs.shape}  |  goal_dist: {info['goal_dist']:.2f}m")

total_r = 0
for _ in range(100):
    obs, r, terminated, truncated, info = env.step(env.action_space.sample())
    total_r += r
    if terminated or truncated:
        obs, info = env.reset()

print(f"100 steps OK  |  cumulative reward: {total_r:.2f}")
print("✓ Environment sanity check passed")
env.close()


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELL 5 — Configure and launch training                                  │
# │                                                                         │
# │ Change MODE and SEED to run different paper experiments.                │
# │                                                                         │
# │ Paper experiment matrix:                                                │
# │   MODE="none"       × SEED=0,1,2  → Naive SAC baseline                 │
# │   MODE="uniform"    × SEED=0,1,2  → Uniform DR baseline                │
# │   MODE="curriculum" × SEED=0,1,2  → CDR (ours) ← start here           │
# │                                                                         │
# │ For ablations (Week 10), use MODE="uniform" with modified               │
# │ auv_dr_wrapper.py that only randomises one parameter at a time.         │
# └─────────────────────────────────────────────────────────────────────────┘

import sys, argparse
sys.path.insert(0, str(REPO_DIR / "scripts"))
sys.path.insert(0, str(REPO_DIR / "envs"))

# ── CONFIGURE YOUR RUN HERE ──────────────────────────────────────────────
MODE  = "curriculum"   # "none" | "uniform" | "curriculum"
SEED  = 0              # 0, 1, or 2 for paper results
STEPS = 1_000_000      # 1M for paper, 50_000 for debug
# ─────────────────────────────────────────────────────────────────────────

from train import train, build_parser

# Build args namespace (same as CLI argparse.Namespace)
args = argparse.Namespace(
    mode     = MODE,
    seed     = SEED,
    steps    = STEPS,
    xml      = str(REPO_DIR / "envs" / "auv.xml"),
    run_name = f"{MODE}_seed{SEED}",
    save_dir = DRIVE_ROOT,
)

print(f"\n{'='*55}")
print(f"  Starting: mode={MODE}  seed={SEED}  steps={STEPS:,}")
print(f"  Save dir: {DRIVE_ROOT}/{MODE}/{MODE}_seed{SEED}")
print(f"{'='*55}\n")

save_dir = train(args)
print(f"\n✓ Training complete → {save_dir}")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELL 6 — TensorBoard (run in Colab during or after training)            │
# └─────────────────────────────────────────────────────────────────────────┘

# Load TensorBoard extension
# %load_ext tensorboard

# Point to Drive log directory
# %tensorboard --logdir /content/drive/MyDrive/rl_research/auv


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELL 7 — Quick eval: load best model and run 20 episodes                │
# │ Use this to sanity-check results before full eval.py                    │
# └─────────────────────────────────────────────────────────────────────────┘

import numpy as np
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ── CONFIGURE ────────────────────────────────────────────────────────────
EVAL_MODE = MODE   # match your training run
EVAL_SEED = SEED
N_EVAL    = 20
# ─────────────────────────────────────────────────────────────────────────

run_dir = Path(DRIVE_ROOT) / EVAL_MODE / f"{EVAL_MODE}_seed{EVAL_SEED}"
model_path = run_dir / "best_model.zip"
stats_path  = run_dir / "vec_normalize.pkl"

if not model_path.exists():
    print(f"✗ No model found at {model_path}")
    print("  Run Cell 5 first to train a model.")
else:
    print(f"Loading model: {model_path}")

    # Rebuild eval env
    def make_eval():
        from auv_env        import HalcyonAUVEnv
        from auv_dr_wrapper import AUVDomainRandomWrapper
        env = HalcyonAUVEnv(xml_path=str(REPO_DIR / "envs" / "auv.xml"))
        env = AUVDomainRandomWrapper(env, mode=EVAL_MODE, seed=EVAL_SEED+999, verbose=False)
        return env

    eval_vec = DummyVecEnv([make_eval])
    eval_vec = VecNormalize.load(str(stats_path), eval_vec)
    eval_vec.training    = False
    eval_vec.norm_reward = False

    # Load model
    model = SAC.load(str(model_path), env=eval_vec, device="cuda")

    # Run eval episodes
    successes, rewards, dists = [], [], []
    obs = eval_vec.reset()
    ep_reward = 0
    ep_count  = 0

    while ep_count < N_EVAL:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_vec.step(action)
        ep_reward += reward[0]
        if done[0]:
            goal_dist = info[0].get("goal_dist", float("inf"))
            success   = goal_dist < 0.5
            successes.append(success)
            rewards.append(ep_reward)
            dists.append(goal_dist)
            ep_reward = 0
            ep_count += 1

    eval_vec.close()

    print(f"\n{'='*45}")
    print(f"  Quick eval — mode={EVAL_MODE}  seed={EVAL_SEED}")
    print(f"  Episodes:     {N_EVAL}")
    print(f"  Success rate: {np.mean(successes):.1%}")
    print(f"  Mean reward:  {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean dist:    {np.mean(dists):.2f}m ± {np.std(dists):.2f}m")
    print(f"{'='*45}")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELL 8 — Run all 9 training conditions (full paper experiment matrix)   │
# │ WARNING: This takes ~18 GPU-hours. Run on Pro+ A100.                    │
# │ Better to run one condition per session to avoid Colab timeouts.        │
# └─────────────────────────────────────────────────────────────────────────┘

# CONDITIONS = [
#     ("none",       0), ("none",       1), ("none",       2),
#     ("uniform",    0), ("uniform",    1), ("uniform",    2),
#     ("curriculum", 0), ("curriculum", 1), ("curriculum", 2),
# ]
#
# for mode, seed in CONDITIONS:
#     print(f"\n{'='*55}")
#     print(f"  Running: mode={mode}  seed={seed}")
#     print(f"{'='*55}")
#     args = argparse.Namespace(
#         mode=mode, seed=seed, steps=1_000_000,
#         xml=str(REPO_DIR / "envs" / "auv.xml"),
#         run_name=f"{mode}_seed{seed}",
#         save_dir=DRIVE_ROOT,
#     )
#     train(args)
#     print(f"✓ Done: {mode} seed={seed}")
#
# print("\n✓ All 9 conditions complete")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELL 9 — CDR curriculum state inspection                                │
# │ Run mid-training to check curriculum progression.                       │
# └─────────────────────────────────────────────────────────────────────────┘

import json
from pathlib import Path

run_dir   = Path(DRIVE_ROOT) / "curriculum" / "curriculum_seed0"
cdr_files = sorted(run_dir.glob("cdr_state_*.json"))

if not cdr_files:
    print("No CDR state files found yet. Run training first.")
else:
    print(f"Found {len(cdr_files)} CDR state snapshots:\n")
    print(f"{'Step':>10}  {'Level':>8}  {'Episodes':>10}  {'Successes':>10}")
    print("-" * 50)
    for f in cdr_files:
        state = json.load(open(f))
        step  = int(f.stem.split("_")[-1])
        print(
            f"{step:>10,}  "
            f"{state['curriculum_level']:>8.3f}  "
            f"{state['n_episodes']:>10,}  "
            f"{state['n_successes']:>10,}"
        )

    # Show final ranges
    if cdr_files:
        final = json.load(open(cdr_files[-1]))
        print(f"\nFinal CDR ranges (step {int(cdr_files[-1].stem.split('_')[-1]):,}):")
        for k, (lo, hi) in final["cdr_ranges"].items():
            print(f"  {k:22s}: [{lo:.4f}, {hi:.4f}]")
