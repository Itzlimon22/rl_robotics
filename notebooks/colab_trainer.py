# ══════════════════════════════════════════════════════════════════
#  colab_trainer.py — Copy these cells into a Colab notebook
#  Your models auto-save to Google Drive. Access from Mac anytime.
# ══════════════════════════════════════════════════════════════════

# ── Cell 1: Mount Drive ───────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')
# Models will save to: MyDrive/rl_research/<env>/<run_name>/

# ── Cell 2: Install deps ──────────────────────────────────────────
# !pip install mujoco gymnasium[mujoco] stable-baselines3[extra] tensorboard -q

# ── Cell 3: Pull your code from GitHub ───────────────────────────
# !git clone https://github.com/YOUR_USERNAME/rl_robotics.git
# %cd rl_robotics

# Or if already cloned, just pull latest:
# !git pull origin main

# ── Cell 4: Check GPU ─────────────────────────────────────────────
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")

# ── Cell 5: Train (saves to Drive automatically) ──────────────────
# !python scripts/train.py \
#     --env HalfCheetah-v4 \
#     --algo sac \
#     --run-name run_01 \
#     --total-steps 1000000 \
#     --n-envs 4

# Swap envs easily:
# --env Hopper-v4
# --env Ant-v4
# --env Humanoid-v4

# ── Cell 6: TensorBoard (live during training) ────────────────────
# %load_ext tensorboard
# %tensorboard --logdir /content/drive/MyDrive/rl_research/HalfCheetah-v4/run_01/tb_logs

# ── Cell 7: Evaluate best model ───────────────────────────────────
# !python scripts/eval.py \
#     --env HalfCheetah-v4 \
#     --run-name run_01 \
#     --n-episodes 10

# ── Cell 8: Record video ──────────────────────────────────────────
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, make_vec_env
from gymnasium.wrappers import RecordVideo
import os

RUN_DIR = "/content/drive/MyDrive/rl_research/HalfCheetah-v4/run_01"
VIDEO_DIR = os.path.join(RUN_DIR, "videos")

env = RecordVideo(
    gym.make("HalfCheetah-v4", render_mode="rgb_array"),
    video_folder=VIDEO_DIR,
    episode_trigger=lambda e: True,
)
model = SAC.load(os.path.join(RUN_DIR, "best_model/best_model"))

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, term, trunc, _ = env.step(action)
    done = term or trunc
env.close()
print(f"Video saved to {VIDEO_DIR}")
