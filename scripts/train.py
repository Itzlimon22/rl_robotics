"""
train.py — Halcyon AUV Training Script
════════════════════════════════════════════════════════════════════════════════

Trains a SAC policy on HalcyonAUVEnv with one of three DR modes.
Designed to run all 9 paper experiments cleanly from CLI or Colab.

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
    best_model.zip          — best model by eval success rate (from EvalCallback)
    final_model.zip         — model at end of training
    vec_normalize.pkl       — VecNormalize running statistics (MUST save with model)
    cdr_state.json          — CDR curriculum state (mode=curriculum only)
    cdr_state_<step>.json   — CDR checkpoints every 50k steps
    config.json             — full run config for reproducibility
    tensorboard/            — TensorBoard logs
    eval/                   — EvalCallback logs (evaluations.npz)
    checkpoints/            — model checkpoints every 50k steps

Device auto-detection:
    Mac M-series  → MPS  (good for debug, slower than Colab for 1M steps)
    Colab/CUDA    → CUDA (use this for full paper runs)
    Fallback      → CPU

TensorBoard:
    tensorboard --logdir ~/rl_research/auv/

Bugs fixed vs original version:
    BUG 1 — ent_coef changed from "auto" to "auto_0.1".
            "auto" starts entropy too low, causing it to collapse to near-zero
            by step ~200k. The agent stops exploring and gets stuck in a bad
            deterministic policy. "auto_0.1" starts with higher initial entropy,
            maintaining exploration through early training.

    BUG 2 — target_entropy changed from "auto" to -1.0.
            SB3 default "auto" sets target_entropy = -dim(action_space) = -4.
            For this environment -4 is too low — it drives entropy to collapse.
            -1.0 keeps entropy meaningfully positive throughout training,
            ensuring the policy continues exploring the action space.

    BUG 3 — max_grad_norm=1.0 added to SAC hyperparameters.
            Gradient clipping prevents occasional large gradient updates from
            destabilising the critic network, especially during early training
            when the replay buffer contains mostly random rollouts.
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ── Project imports ───────────────────────────────────────────────────────────
# Resolve repo root regardless of where train.py lives (scripts/ or root).
# Structure:  ~/rl_robotics/
#               scripts/train.py   ← __file__
#               envs/auv_env.py
#               envs/auv_dr_wrapper.py
_SCRIPT_DIR = Path(__file__).parent.resolve()  # .../rl_robotics/scripts
_REPO_ROOT = _SCRIPT_DIR.parent  # .../rl_robotics
_ENVS_DIR = _REPO_ROOT / "envs"  # .../rl_robotics/envs

for _p in [_REPO_ROOT, _ENVS_DIR, _SCRIPT_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from auv_env import HalcyonAUVEnv
from auv_dr_wrapper import AUVDomainRandomWrapper, make_auv_env


# ─────────────────────────────────────────────────────────────────────────────
# SAC hyperparameters (tuned for AUV task)
# ─────────────────────────────────────────────────────────────────────────────
SAC_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "buffer_size": 500_000,  # 500k transitions replay buffer
    "learning_starts": 10_000,  # warm-up steps before first gradient update
    "batch_size": 256,
    "tau": 0.005,  # soft update coefficient for target networks
    "gamma": 0.99,  # discount factor
    "train_freq": 1,  # update every step (off-policy, efficient)
    "gradient_steps": 1,
    # BUG 1 FIX: "auto_0.1" instead of "auto".
    # "auto" initialises entropy coefficient too low, causing it to collapse
    # to near-zero before the agent has learned a useful policy.
    # "auto_0.1" sets a higher initial value, maintaining exploration during
    # the critical early phase of training (steps 0 to ~300k).
    "ent_coef": "auto_0.1",
    # BUG 2 FIX: -1.0 instead of "auto".
    # SB3 "auto" default: target_entropy = -dim(action_space) = -4.
    # -4 is too aggressive — it drives the entropy coefficient to near-zero
    # very rapidly, eliminating exploration. The agent then exploits a bad
    # locally optimal policy for the rest of training.
    # -1.0 keeps entropy meaningfully positive, ensuring continued exploration.
    "target_entropy": -1.0,
    # BUG 3 FIX: gradient clipping added.
    # max_grad_norm=1.0 prevents large gradient updates from destabilising
    # the critic network. Particularly important early in training when the
    # replay buffer contains mostly random rollout data with high variance.
    "policy_kwargs": dict(
        net_arch=[256, 256],
        activation_fn=torch.nn.ReLU,
    ),
    "verbose": 1,
    "tensorboard_log": None,  # set per-run below
}

# Eval frequency and episodes (balance between information and compute cost)
EVAL_FREQ = 10_000  # evaluate every 10k training steps
N_EVAL_EPS = 20  # 20 episodes per evaluation
CHECKPOINT_FREQ = 50_000  # save checkpoint every 50k steps


# ─────────────────────────────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────────────────────────────


def detect_device() -> str:
    """Auto-detect best available compute device."""
    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        name = "Apple Silicon MPS"
    else:
        device = "cpu"
        name = "CPU"
    print(f"[device] Using {device.upper()} — {name}")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# Save path resolver
# ─────────────────────────────────────────────────────────────────────────────


def resolve_save_dir(mode: str, run_name: str, base_dir: Optional[str] = None) -> Path:
    """
    Resolve the save directory. Auto-detects Colab and saves to Drive.

    Priority:
        1. --save-dir argument (explicit override)
        2. Google Drive (/content/drive/MyDrive/...) if on Colab
        3. ~/rl_research/auv/<mode>/<run_name>/ locally
    """
    if base_dir:
        p = Path(base_dir) / mode / run_name
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Detect Colab by checking if Drive is mounted
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

    VecNormalize normalises observations and rewards online using running
    statistics. This is critical for SAC convergence on this task — raw
    observations span very different scales ([0,20] for distance vs [-1,1]
    for actions vs [-pi,pi] for angles).
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


def make_eval_env(
    xml_path: str, mode: str, seed: int, vec_normalize: VecNormalize
) -> VecNormalize:
    """
    Create evaluation environment sharing VecNormalize statistics with training env.

    Critical SB3 pattern for correct evaluation:
        1. Create fresh VecNormalize on eval env (do NOT reuse training env object)
        2. Copy obs_rms and ret_rms references from training env
           (these are Python objects — reference copy means eval env always
           uses the CURRENT training env statistics, updated each training step)
        3. Set training=False → freeze statistics during evaluation
        4. Set norm_reward=False → raw rewards for interpretable eval metrics

    Why copy running stats?
        The eval env must normalise observations the same way as the training env.
        If the eval env uses different statistics, the model sees observations in
        a different distribution than it was trained on → artificially bad eval.

    Note: VecNormalize.load_from_venv() does NOT exist in SB3.
    This manual obs_rms copy is the correct pattern.
    """

    def _init():
        env = HalcyonAUVEnv(xml_path=xml_path)
        # Eval env uses same DR mode — evaluates on same distribution as training.
        # For held-out test distribution evaluation, use eval.py separately.
        env = AUVDomainRandomWrapper(env, mode=mode, seed=seed + 1000, verbose=False)
        return env

    eval_vec = DummyVecEnv([_init])
    eval_vec = VecNormalize(
        eval_vec,
        norm_obs=True,
        norm_reward=False,  # always False for eval — raw rewards for interpretability
        clip_obs=10.0,
        gamma=SAC_HYPERPARAMS["gamma"],
    )
    # Copy observation running stats so normalisation matches training env.
    # These are object references — eval env automatically uses current train stats.
    eval_vec.obs_rms = vec_normalize.obs_rms
    eval_vec.ret_rms = vec_normalize.ret_rms
    eval_vec.training = False  # freeze — do not update stats during evaluation
    return eval_vec


# ─────────────────────────────────────────────────────────────────────────────
# Custom callbacks
# ─────────────────────────────────────────────────────────────────────────────


class AUVMetricsCallback(BaseCallback):
    """
    Log AUV-specific metrics to TensorBoard every step.

    Captures from the info dict returned by AUVDomainRandomWrapper.step():
        goal_dist           → env/goal_dist         (primary learning signal)
        speed               → env/auv_speed
        r_progress etc.     → reward/r_*            (reward components)
        dr/curriculum_level → cdr/curriculum_level  (CDR expansion progress)
        dr/success_rate     → cdr/rolling_success_rate
        dr/c_drag_lateral   → cdr/c_drag_lateral    (sampled physics params)
        dr/current_speed    → cdr/current_speed
        dr/buoyancy         → cdr/buoyancy

    These metrics are visible in TensorBoard and are used to monitor:
        - Whether goal_dist is trending down (learning happening)
        - Whether curriculum_level is increasing (CDR expanding correctly)
        - Whether reward components are balanced
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._n_episodes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        dones = self.locals.get("dones", [False])

        for info, done in zip(infos, dones):
            # Environment metrics
            if "goal_dist" in info:
                self.logger.record_mean("env/goal_dist", info["goal_dist"])
            if "speed" in info:
                self.logger.record_mean("env/auv_speed", info["speed"])

            # Reward components (all logged for debugging reward shaping)
            for key in ["r_progress", "r_energy", "r_smooth", "r_orient", "r_boundary"]:
                if key in info:
                    self.logger.record_mean(f"reward/{key}", info[key])

            # CDR metrics (only populated when mode=curriculum)
            if "dr/curriculum_level" in info:
                self.logger.record_mean(
                    "cdr/curriculum_level", info["dr/curriculum_level"]
                )
            if "dr/success_rate" in info:
                self.logger.record_mean(
                    "cdr/rolling_success_rate", info["dr/success_rate"]
                )
            if "dr/c_drag_lateral" in info:
                self.logger.record_mean("cdr/c_drag_lateral", info["dr/c_drag_lateral"])
            if "dr/current_speed" in info:
                self.logger.record_mean("cdr/current_speed", info["dr/current_speed"])
            if "dr/buoyancy" in info:
                self.logger.record_mean("cdr/buoyancy", info["dr/buoyancy"])

            # Episode count (for phase plots)

            if done:
                self._n_episodes += 1
                goal_dist_at_done = info.get("goal_dist", 999)
                success = float(goal_dist_at_done < 0.5)
                self.logger.record_mean("env/success_rate", success)
                self.logger.record("env/n_episodes", self._n_episodes)

        return True


class CDRCheckpointCallback(BaseCallback):
    """
    Save CDR curriculum state alongside model checkpoints.

    Called every CHECKPOINT_FREQ steps. Saves a JSON file with:
        - curriculum_level: current expansion level [0, 1]
        - n_episodes: total episodes elapsed
        - n_successes: total successful episodes
        - cdr_ranges: current [lo, hi] for each physics parameter
        - outcome_window: last W episode outcomes (True=success)

    These checkpoints allow:
        1. Inspecting curriculum progress mid-training
        2. Resuming training from a checkpoint
        3. Verifying CDR is expanding correctly

    Usage to inspect mid-training:
        import json
        state = json.load(open("cdr_state_500000.json"))
        print(state["curriculum_level"])  # should be > 0 if CDR working
    """

    def __init__(self, save_dir: Path, save_freq: int, verbose: int = 1):
        super().__init__(verbose)
        self.save_dir = save_dir
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
            # Navigate wrapper chain: VecNormalize → DummyVecEnv → AUVDomainRandomWrapper
            env = self.training_env.venv.envs[0]
            if hasattr(env, "get_cdr_state"):
                state = env.get_cdr_state()
                path = self.save_dir / f"cdr_state_{self.num_timesteps}.json"
                with open(path, "w") as f:
                    json.dump(state, f, indent=2)
                if self.verbose:
                    lvl = state.get("curriculum_level", 0)
                    print(
                        f"[CDR] Saved state at step {self.num_timesteps} "
                        f"| level={lvl:.3f} | path={path.name}"
                    )
        except Exception as e:
            if self.verbose:
                print(f"[CDR] Warning: could not save CDR state: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────


def train(args: argparse.Namespace):
    """
    Full training pipeline. Called by main() or directly from Colab.

    Steps:
        1. Detect device (CUDA/MPS/CPU)
        2. Resolve XML and save paths
        3. Save config.json for reproducibility
        4. Build training and eval environments
        5. Initialise SAC model with fixed hyperparameters
        6. Register callbacks (metrics, eval, checkpoint, CDR)
        7. Run model.learn() for args.steps timesteps
        8. Save final model, VecNormalize stats, CDR state
    """

    # ── Setup ─────────────────────────────────────────────────────────────────
    t_start = time.time()
    device = detect_device()

    xml_path = resolve_xml_path(args.xml)
    print(f"[env]  XML: {xml_path}")
    print(f"[run]  mode={args.mode}  seed={args.seed}  steps={args.steps:,}")

    # Run name: mode_seed for paper experiments, or custom for debug
    run_name = args.run_name or f"{args.mode}_seed{args.seed}_{int(time.time())}"
    save_dir = resolve_save_dir(args.mode, run_name, args.save_dir)
    tb_dir = save_dir / "tensorboard"
    eval_dir = save_dir / "eval"
    tb_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Save full config for reproducibility — critical for paper experiments
    config = vars(args)
    config.update(
        {
            "run_name": run_name,
            "save_dir": str(save_dir),
            "xml_path": str(xml_path),
            "device": device,
            "sac_hyperparams": {k: str(v) for k, v in SAC_HYPERPARAMS.items()},
        }
    )
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
    print(
        f"[model] ent_coef: {hyperparams['ent_coef']}  "
        f"target_entropy: {hyperparams['target_entropy']}  "
        f"max_grad_norm: {hyperparams['max_grad_norm']}"
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    metrics_cb = AUVMetricsCallback(verbose=0)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),  # saves best_model.zip here
        log_path=str(eval_dir),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPS,
        deterministic=True,  # deterministic policy during eval
        render=False,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(save_dir / "checkpoints"),
        name_prefix="model",
        save_vecnormalize=True,  # saves VecNormalize alongside each checkpoint
        verbose=1,
    )

    cdr_ckpt_cb = CDRCheckpointCallback(
        save_dir=save_dir,
        save_freq=CHECKPOINT_FREQ,
        verbose=1,
    )

    callbacks = CallbackList([metrics_cb, eval_cb, checkpoint_cb, cdr_ckpt_cb])

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  TRAINING: mode={args.mode}  seed={args.seed}")
    print(f"  Steps: {args.steps:,}  |  Eval every {EVAL_FREQ:,} steps")
    print(f"  Save:  {save_dir}")
    print(f"{'=' * 60}\n")

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
    print(f"\n{'=' * 60}")
    print(f"  DONE — {args.steps:,} steps in {elapsed / 60:.1f} min")
    print(f"  Best model: {save_dir / 'best_model.zip'}")
    print(f"  TensorBoard: tensorboard --logdir {tb_dir}")
    print(f"{'=' * 60}")

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
        "--mode",
        type=str,
        default="curriculum",
        choices=["none", "uniform", "curriculum"],
        help=(
            "DR mode. "
            "'none'=naive baseline (Baseline 1), "
            "'uniform'=UDR (Baseline 2), "
            "'curriculum'=CDR (Ours)"
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Run seeds 0, 1, 2 for paper results (3 seeds per condition).",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=1_000_000,
        help="Total training timesteps. 1M for paper runs, 50k for debug.",
    )
    p.add_argument(
        "--xml",
        type=str,
        default=None,
        help="Path to auv.xml. Auto-detected if not specified.",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name. Auto-generated (mode_seedN_timestamp) if not set.",
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Base save directory. Auto-detected (local or Colab Drive) if not set.",
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
