"""
train.py — Halcyon AUV Training Script
════════════════════════════════════════════════════════════════════════════════

Trains a SAC policy on HalcyonAUVEnv with one of three DR modes.
Designed to run all 24 paper experiments cleanly from CLI or Colab.

Usage (local Mac, debug run):
    python train.py --mode curriculum --seed 42 --steps 50000 --run-name debug

Usage (full paper experiment):
    python train.py --mode curriculum --seed 0 --steps 1000000
    python train.py --mode uniform    --seed 0 --steps 1000000
    python train.py --mode none       --seed 0 --steps 1000000

Run all 3 seeds for one condition (bash):
    for seed in 0 1 2; do
        python train.py --mode curriculum --seed $seed --steps 1000000
    done

Saves to:
    Local:  ~/rl_research/auv/<mode>/<run_name>/
    Colab:  /content/drive/MyDrive/rl_research/auv/<mode>/<run_name>/

Each run directory contains:
    best_model.zip          — best model by eval success rate
    final_model.zip         — model at end of training
    vec_normalize.pkl       — VecNormalize running statistics (MUST save with model)
    cdr_state.json          — CDR curriculum state (mode=curriculum only)
    config.json             — full run config for reproducibility
    tensorboard/            — TensorBoard logs
    eval/                   — EvalCallback logs (evaluations.npz)

Device auto-detection:
    Mac M-series  → MPS  (fast for debug, slow for 1M steps vs Colab)
    Colab/CUDA    → CUDA (use this for full runs)
    Fallback      → CPU

TensorBoard:
    tensorboard --logdir ~/rl_research/auv/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ── SB3 imports ──────────────────────────────────────────────────────────────
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ── Project imports ───────────────────────────────────────────────────────────
# Add envs/ to path so imports work from any working directory
_SCRIPT_DIR = Path(__file__).parent.resolve()
_ENVS_DIR   = _SCRIPT_DIR / "envs"
for _p in [_SCRIPT_DIR, _ENVS_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from auv_env       import HalcyonAUVEnv
from auv_dr_wrapper import AUVDomainRandomWrapper, make_auv_env


# ─────────────────────────────────────────────────────────────────────────────
# SAC hyperparameters (tuned for AUV task)
# These are locked in after the Week 6 hyperparameter sweep.
# Change only if sweep results suggest otherwise.
# ─────────────────────────────────────────────────────────────────────────────
SAC_HYPERPARAMS = {
    "learning_rate":         3e-4,
    "buffer_size":           500_000,    # 500k transitions replay buffer
    "learning_starts":       10_000,     # warm-up steps before first update
    "batch_size":            256,
    "tau":                   0.005,      # soft update coefficient
    "gamma":                 0.99,       # discount factor
    "train_freq":            1,          # update every step (off-policy)
    "gradient_steps":        1,
    "ent_coef":              "auto",     # automatic entropy tuning (SAC hallmark)
    "target_entropy":        "auto",
    "policy_kwargs":         dict(
        net_arch=[256, 256],             # 2-layer MLP, 256 units each
        activation_fn=torch.nn.ReLU,
    ),
    "verbose":               1,
    "tensorboard_log":       None,       # set per-run below
}

# Eval frequency and episodes (balance between info and compute cost)
EVAL_FREQ      = 10_000   # evaluate every 10k training steps
N_EVAL_EPS     = 20       # 20 episodes per evaluation
CHECKPOINT_FREQ = 50_000  # save checkpoint every 50k steps


# ─────────────────────────────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        name   = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        name   = "Apple Silicon MPS"
    else:
        device = "cpu"
        name   = "CPU"
    print(f"[device] Using {device.upper()} — {name}")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# Save path resolver
# ─────────────────────────────────────────────────────────────────────────────

def resolve_save_dir(mode: str, run_name: str, base_dir: Optional[str] = None) -> Path:
    """
    Resolve the save directory. Auto-detects Colab and saves to Drive.

    Priority:
        1. --save-dir argument (explicit)
        2. Google Drive (/content/drive/MyDrive/...) if on Colab
        3. ~/rl_research/auv/<mode>/<run_name>/
    """
    if base_dir:
        return Path(base_dir) / mode / run_name

    # Detect Colab
    on_colab = os.path.exists("/content/drive/MyDrive")
    if on_colab:
        drive_path = Path("/content/drive/MyDrive/rl_research/auv") / mode / run_name
        drive_path.mkdir(parents=True, exist_ok=True)
        print(f"[save] Colab detected → saving to Drive: {drive_path}")
        return drive_path

    local_path = Path.home() / "rl_research" / "auv" / mode / run_name
    local_path.mkdir(parents=True, exist_ok=True)
    print(f"[save] Local save → {local_path}")
    return local_path


def resolve_xml_path(xml_arg: Optional[str]) -> Path:
    """Find auv.xml in standard locations."""
    candidates = []
    if xml_arg:
        candidates.append(Path(xml_arg))
    candidates += [
        _ENVS_DIR / "auv.xml",
        _SCRIPT_DIR / "auv.xml",
        Path.home() / "rl_robotics" / "envs" / "auv.xml",
        Path("auv.xml"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "auv.xml not found. Pass --xml path/to/auv.xml or place it in envs/auv.xml"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def make_train_env(xml_path: str, mode: str, seed: int) -> VecNormalize:
    """
    Create the VecNormalize-wrapped training environment.

    Stack: HalcyonAUVEnv → AUVDomainRandomWrapper → DummyVecEnv → VecNormalize
    """
    def _init():
        env = HalcyonAUVEnv(xml_path=xml_path)
        env = AUVDomainRandomWrapper(env, mode=mode, seed=seed, verbose=True)
        return env

    vec_env = DummyVecEnv([_init])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=SAC_HYPERPARAMS["gamma"],
    )
    return vec_env


def make_eval_env(xml_path: str, mode: str, seed: int, vec_normalize: VecNormalize) -> VecNormalize:
    """
    Create evaluation environment sharing VecNormalize statistics with training env.

    Critical SB3 pattern:
        - eval env uses same running stats as train env (do not recompute)
        - vec_env.training = False → stats frozen during eval
        - vec_env.norm_reward = False → raw rewards for interpretable eval metrics
    """
    def _init():
        env = HalcyonAUVEnv(xml_path=xml_path)
        # Eval env uses same DR mode — evaluates on same distribution as training
        # For held-out test evaluation, use eval.py with test distribution
        env = AUVDomainRandomWrapper(env, mode=mode, seed=seed + 1000, verbose=False)
        return env

    eval_vec = DummyVecEnv([_init])
    # Load running stats from training env
    eval_vec = VecNormalize.load_from_venv(vec_normalize, eval_vec)  # shares stats
    eval_vec.training   = False   # freeze running mean/std
    eval_vec.norm_reward = False  # raw rewards for EvalCallback mean_reward metric
    return eval_vec


# ─────────────────────────────────────────────────────────────────────────────
# Custom callbacks
# ─────────────────────────────────────────────────────────────────────────────

class AUVMetricsCallback(BaseCallback):
    """
    Log AUV-specific metrics to TensorBoard every episode.

    Captures from the info dict:
        - goal_dist           → env/goal_dist
        - r_progress etc.     → reward/component_*
        - dr/curriculum_level → cdr/curriculum_level
        - dr/success_rate     → cdr/rolling_success_rate
        - dr/c_drag_lateral   → cdr/c_drag_lateral
        - dr/current_speed    → cdr/current_speed
        - speed               → env/auv_speed
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_rewards: list = []
        self._episode_lengths: list = []
        self._n_episodes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        dones = self.locals.get("dones", [False])

        for info, done in zip(infos, dones):
            # Log step-level metrics
            if "goal_dist" in info:
                self.logger.record_mean("env/goal_dist", info["goal_dist"])
            if "speed" in info:
                self.logger.record_mean("env/auv_speed", info["speed"])

            # Reward components
            for key in ["r_progress", "r_energy", "r_smooth", "r_orient", "r_boundary"]:
                if key in info:
                    self.logger.record_mean(f"reward/{key}", info[key])

            # CDR metrics (only populated in curriculum mode)
            if "dr/curriculum_level" in info:
                self.logger.record_mean("cdr/curriculum_level", info["dr/curriculum_level"])
            if "dr/success_rate" in info:
                self.logger.record_mean("cdr/rolling_success_rate", info["dr/success_rate"])
            if "dr/c_drag_lateral" in info:
                self.logger.record_mean("cdr/c_drag_lateral", info["dr/c_drag_lateral"])
            if "dr/current_speed" in info:
                self.logger.record_mean("cdr/current_speed", info["dr/current_speed"])
            if "dr/buoyancy" in info:
                self.logger.record_mean("cdr/buoyancy", info["dr/buoyancy"])

            # Episode-level (only on done)
            if done and "episode" in info:
                self._n_episodes += 1
                self.logger.record("env/n_episodes", self._n_episodes)

        return True


class CDRCheckpointCallback(BaseCallback):
    """
    Save CDR curriculum state alongside model checkpoints.
    Called every CHECKPOINT_FREQ steps.
    """

    def __init__(self, save_dir: Path, save_freq: int, verbose: int = 1):
        super().__init__(verbose)
        self.save_dir  = save_dir
        self.save_freq = save_freq
        self._last_save = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self.save_freq:
            self._save_cdr_state()
            self._last_save = self.num_timesteps
        return True

    def _save_cdr_state(self):
        """Extract CDR state from env wrapper chain and save to JSON."""
        try:
            # Navigate: VecNormalize → DummyVecEnv → AUVDomainRandomWrapper
            env = self.training_env.venv.envs[0]  # unwrap DummyVecEnv
            if hasattr(env, "get_cdr_state"):
                state = env.get_cdr_state()
                path  = self.save_dir / f"cdr_state_{self.num_timesteps}.json"
                with open(path, "w") as f:
                    json.dump(state, f, indent=2)
                if self.verbose:
                    lvl = state.get("curriculum_level", 0)
                    print(f"[CDR] Saved state at step {self.num_timesteps} "
                          f"| level={lvl:.3f} | path={path.name}")
        except Exception as e:
            if self.verbose:
                print(f"[CDR] Warning: could not save CDR state: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace):
    """Full training pipeline. Called by main() or Colab."""

    # ── Setup ─────────────────────────────────────────────────────────────────
    t_start = time.time()
    device  = detect_device()

    xml_path = resolve_xml_path(args.xml)
    print(f"[env]  XML: {xml_path}")
    print(f"[run]  mode={args.mode}  seed={args.seed}  steps={args.steps:,}")

    # Run name: mode_seed_timestamp for uniqueness
    run_name = args.run_name or f"{args.mode}_seed{args.seed}_{int(time.time())}"
    save_dir = resolve_save_dir(args.mode, run_name, args.save_dir)
    tb_dir   = save_dir / "tensorboard"
    eval_dir = save_dir / "eval"
    tb_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Save full config for reproducibility
    config = vars(args)
    config.update({
        "run_name":   run_name,
        "save_dir":   str(save_dir),
        "xml_path":   str(xml_path),
        "device":     device,
        "sac_hyperparams": {k: str(v) for k, v in SAC_HYPERPARAMS.items()},
    })
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[run]  Config saved → {save_dir / 'config.json'}")

    # ── Environments ──────────────────────────────────────────────────────────
    print("\n[env]  Building training environment...")
    train_env = make_train_env(str(xml_path), args.mode, args.seed)

    print("[env]  Building eval environment...")
    eval_env = make_eval_env(str(xml_path), args.mode, args.seed, train_env)

    # ── Model ─────────────────────────────────────────────────────────────────
    hyperparams = dict(SAC_HYPERPARAMS)
    hyperparams["tensorboard_log"] = str(tb_dir)

    print("\n[model] Initialising SAC...")
    model = SAC(
        "MlpPolicy",
        train_env,
        device=device,
        seed=args.seed,
        **hyperparams,
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"[model] Policy parameters: {n_params:,}")
    print(f"[model] Network: MLP {hyperparams['policy_kwargs']['net_arch']}")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    metrics_cb = AUVMetricsCallback(verbose=0)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(eval_dir),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPS,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(save_dir / "checkpoints"),
        name_prefix="model",
        save_vecnormalize=True,  # saves VecNormalize alongside checkpoint
        verbose=1,
    )

    cdr_ckpt_cb = CDRCheckpointCallback(
        save_dir=save_dir,
        save_freq=CHECKPOINT_FREQ,
        verbose=1,
    )

    callbacks = CallbackList([metrics_cb, eval_cb, checkpoint_cb, cdr_ckpt_cb])

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TRAINING: mode={args.mode}  seed={args.seed}")
    print(f"  Steps: {args.steps:,}  |  Eval every {EVAL_FREQ:,} steps")
    print(f"  Save:  {save_dir}")
    print(f"{'='*60}\n")

    try:
        model.learn(
            total_timesteps=args.steps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=True,
            tb_log_name=run_name,
        )
    except KeyboardInterrupt:
        print("\n[train] Interrupted — saving current state...")

    # ── Save final artifacts ──────────────────────────────────────────────────
    print("\n[save] Saving final model and VecNormalize stats...")
    model.save(str(save_dir / "final_model"))
    train_env.save(str(save_dir / "vec_normalize.pkl"))

    # Save final CDR state
    try:
        env = train_env.venv.envs[0]
        if hasattr(env, "get_cdr_state"):
            cdr_state = env.get_cdr_state()
            with open(save_dir / "cdr_state.json", "w") as f:
                json.dump(cdr_state, f, indent=2)
            print(f"[save] CDR state → {save_dir / 'cdr_state.json'}")
            print(f"       Final curriculum level: {cdr_state['curriculum_level']:.3f}")
            print(f"       Total episodes:         {cdr_state['n_episodes']}")
            print(f"       Total successes:        {cdr_state['n_successes']}")
    except Exception as e:
        print(f"[save] Warning: CDR state not saved: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  DONE — {args.steps:,} steps in {elapsed/60:.1f} min")
    print(f"  Best model: {save_dir / 'best_model.zip'}")
    print(f"  TensorBoard: tensorboard --logdir {tb_dir}")
    print(f"{'='*60}")

    train_env.close()
    eval_env.close()
    return save_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train SAC on Halcyon AUV with domain randomisation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode", type=str, default="curriculum",
        choices=["none", "uniform", "curriculum"],
        help="DR mode. 'none'=naive baseline, 'uniform'=UDR, 'curriculum'=CDR (ours)",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Random seed. Run 0,1,2 for paper results (3 seeds each condition).",
    )
    p.add_argument(
        "--steps", type=int, default=1_000_000,
        help="Total training timesteps. 1M for paper, 50k for debug.",
    )
    p.add_argument(
        "--xml", type=str, default=None,
        help="Path to auv.xml. Auto-detected if not specified.",
    )
    p.add_argument(
        "--run-name", type=str, default=None,
        help="Run name suffix. Auto-generated (mode_seedN_timestamp) if not set.",
    )
    p.add_argument(
        "--save-dir", type=str, default=None,
        help="Base save directory. Auto-detected (local or Colab Drive) if not set.",
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
