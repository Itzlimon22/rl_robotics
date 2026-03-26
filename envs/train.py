"""
train.py — Baseline SAC Training for Halcyon AUV CDR Research
══════════════════════════════════════════════════════════════════════════════

Trains three experimental conditions for the paper:
  Condition A — "Naive SAC"   : no DR, nominal physics only
  Condition B — "Uniform DR"  : full DR range from episode 1
  Condition C — "CDR (ours)"  : curriculum DR, ranges expand with performance

All three share identical SAC hyperparameters, seeds, and total timesteps
so that results are directly comparable in paper figures.

Usage (single condition):
    python train.py --mode curriculum --seed 0 --timesteps 1000000

Usage (all three conditions, sequential):
    python train.py --mode all --seed 0 --timesteps 1000000

Usage (all three conditions, three seeds each — paper Table 2):
    for seed in 0 1 2; do
        for mode in none uniform curriculum; do
            python train.py --mode $mode --seed $seed --timesteps 1000000
        done
    done

Outputs (all under --log-dir, default ~/rl_research/auv/):
    logs/
      {mode}_seed{seed}/          ← TensorBoard logs
        evaluations.npz           ← EvalCallback data (for learning curves)
        best_model.zip            ← best checkpoint by eval reward
    checkpoints/
      {mode}_seed{seed}/          ← model checkpoint every 50k steps
    results/
      {mode}_seed{seed}/
        final_model.zip           ← model at end of training
        cdr_state.json            ← CDR curriculum state (mode=curriculum only)
        eval_transfer.npz         ← transfer eval on held-out test distribution
        training_config.json      ← full config snapshot for reproducibility

Paper figure pipeline:
    After all runs complete, use:
        python plot_results.py --log-dir ~/rl_research/auv/
    to generate:
        Fig 3 — Learning curves (mean ± std across seeds)
        Fig 4 — CDR curriculum level over training
        Fig 5 — Transfer success rate on held-out distribution
        Table 2 — Final performance statistics

Install:
    pip install "gymnasium[mujoco]" "stable-baselines3[extra]" mujoco numpy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from callbacks.render_callback import LiveRenderCallback  # ✅ keep

import numpy as np

# ── SB3 imports ──────────────────────────────────────────────────────────────
try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
except ImportError:
    raise ImportError(
        "stable-baselines3 required.\n"
        "Install with: pip install 'stable-baselines3[extra]'"
    )

# ── Project imports ───────────────────────────────────────────────────────────
# Add project root to path so imports work regardless of working directory
_PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "envs"))

from auv_env import HalcyonAUVEnv
from auv_dr_wrapper import AUVDomainRandomWrapper, make_auv_env


# ══════════════════════════════════════════════════════════════════════════════
# SAC hyperparameters
# Tuned for continuous 4D action, 18D observation AUV goal-reaching.
# Keep IDENTICAL across all three conditions for fair comparison.
# ══════════════════════════════════════════════════════════════════════════════

SAC_HYPERPARAMS: Dict = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,  # Adam lr, standard for SAC
    "buffer_size": 300_000,  # replay buffer capacity
    "learning_starts": 5_000,  # steps before first gradient update
    "batch_size": 256,  # minibatch size for gradient updates
    "tau": 0.005,  # soft target network update rate
    "gamma": 0.99,  # discount factor
    "train_freq": 1,  # update every step
    "gradient_steps": 1,  # gradient updates per env step
    "ent_coef": "auto",  # automatic entropy tuning (key for SAC)
    "target_entropy": "auto",  # -dim(A) = -4 by default
    "use_sde": False,  # State-Dependent Exploration (off for AUV)
    "policy_kwargs": dict(
        net_arch=[256, 256],  # 2-layer MLP, 256 units each
        activation_fn=__import__("torch.nn", fromlist=["ReLU"]).ReLU,
    ),
    "verbose": 0,  # suppress SB3 training output (use TB instead)
}


# ══════════════════════════════════════════════════════════════════════════════
# Training configuration
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG: Dict = {
    "xml_path": str(_PROJECT_ROOT / "envs" / "auv.xml"),
    "total_timesteps": 1_000_000,
    "eval_freq": 10_000,  # evaluate every N training steps
    "n_eval_episodes": 20,  # episodes per evaluation
    "checkpoint_freq": 50_000,  # save checkpoint every N steps
    "log_dir": str(Path.home() / "rl_research" / "auv"),
    "frame_skip": 4,  # effective control dt = 4 × 0.01 = 0.04s
    "max_episode_steps": 500,  # 500 × 0.04s = 20 second episodes
    "goal_threshold": 0.5,  # metres — goal reached radius
    "workspace_radius": 15.0,  # metres — episode abort boundary
}


# ══════════════════════════════════════════════════════════════════════════════
# Custom SB3 callbacks
# ══════════════════════════════════════════════════════════════════════════════


class CDRLoggingCallback(BaseCallback):
    """
    Logs CDR-specific metrics to TensorBoard at every step.

    Logged scalars:
      cdr/curriculum_level   — expansion progress [0, 1]  (CDR mode only)
      cdr/rolling_success    — rolling success rate         (CDR mode only)
      cdr/current_speed      — sampled current this episode
      cdr/c_drag_lateral     — sampled drag this episode
      cdr/n_episodes         — total episodes elapsed
      env/goal_dist          — current goal distance (m)
      env/speed              — current AUV speed (m/s)
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        for info in infos:
            if "dr/curriculum_level" in info:
                self.logger.record("cdr/curriculum_level", info["dr/curriculum_level"])
            if "dr/success_rate" in info:
                self.logger.record("cdr/rolling_success", info["dr/success_rate"])
            if "dr/current_speed" in info:
                self.logger.record("cdr/current_speed", info["dr/current_speed"])
            if "dr/c_drag_lateral" in info:
                self.logger.record("cdr/c_drag_lateral", info["dr/c_drag_lateral"])
            if "dr/n_episodes" in info:
                self.logger.record("cdr/n_episodes", info["dr/n_episodes"])
            if "goal_dist" in info:
                self.logger.record("env/goal_dist", info["goal_dist"])
            if "speed" in info:
                self.logger.record("env/speed", info["speed"])
        return True


class TimingCallback(BaseCallback):
    """
    Logs steps-per-second to TensorBoard.
    Useful for benchmarking simulation throughput across machines.
    """

    def __init__(self, log_every: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self._log_every = log_every
        self._t0 = time.perf_counter()
        self._last_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self._log_every == 0:
            elapsed = time.perf_counter() - self._t0
            steps = self.num_timesteps - self._last_step
            sps = steps / max(elapsed, 1e-6)
            self.logger.record("perf/steps_per_second", sps)
            self._t0 = time.perf_counter()
            self._last_step = self.num_timesteps
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Environment factory
# ══════════════════════════════════════════════════════════════════════════════


def _make_env_fn(
    xml_path: str,
    mode: str,
    seed: int,
    render_mode: Optional[str] = None,
):
    """
    Returns a callable that creates one training env instance.
    Used by make_vec_env / DummyVecEnv.
    """

    def _init():
        env = make_auv_env(
            xml_path=xml_path,
            mode=mode,
            seed=seed,
            render_mode=render_mode,
        )
        return env

    return _init


def build_training_env(cfg: Dict, mode: str, seed: int) -> VecNormalize:
    """
    Build a VecNormalize-wrapped training environment.

    VecNormalize normalises observations (running mean/std) and reward
    (running mean/std). This is standard practice with SAC + SB3 and
    avoids manual reward scaling across conditions.

    The normalisation statistics are saved with the model and restored
    at evaluation time (critical for fair comparison).
    """
    vec_env = DummyVecEnv([_make_env_fn(cfg["xml_path"], mode, seed)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    return vec_env


def build_eval_env(
    cfg: Dict,
    mode: str,
    seed: int,
    vec_normalize: Optional[VecNormalize] = None,
    test_distribution: bool = False,
) -> VecNormalize:
    """
    Build a VecNormalize-wrapped evaluation environment.

    Parameters
    ----------
    cfg : dict
        Training config.
    mode : str
        DR mode for the eval env.
    seed : int
        Eval RNG seed (use seed+1000 to separate from training).
    vec_normalize : VecNormalize | None
        If provided, copy normalisation statistics from the training env.
        This ensures eval observations are in the same normalised space.
    test_distribution : bool
        If True, switch the eval env to the held-out test distribution.
        Used for the transfer evaluation after training.
    """
    eval_vec = DummyVecEnv([_make_env_fn(cfg["xml_path"], mode, seed + 1000)])
    eval_vec = VecNormalize(
        eval_vec,
        norm_obs=True,
        norm_reward=False,  # do NOT normalise reward at eval (we want true reward)
        clip_obs=10.0,
        training=False,  # frozen stats — do not update during eval
    )

    # Sync normalisation stats from training env if available
    if vec_normalize is not None:
        eval_vec.obs_rms = vec_normalize.obs_rms
        eval_vec.ret_rms = vec_normalize.ret_rms

    # Optionally switch to held-out test distribution
    if test_distribution:
        # Unwrap to reach AUVDomainRandomWrapper
        inner = eval_vec.venv.envs[0]
        if isinstance(inner, AUVDomainRandomWrapper):
            inner.set_test_distribution()

    return eval_vec


# ══════════════════════════════════════════════════════════════════════════════
# Main training function
# ══════════════════════════════════════════════════════════════════════════════


def train_condition(
    mode: str,
    seed: int,
    cfg: Dict,
    device: str = "auto",
) -> Dict:
    """
    Train one experimental condition (mode × seed).

    Parameters
    ----------
    mode : str
        One of "none", "uniform", "curriculum".
    seed : int
        Random seed for reproducibility.
    cfg : dict
        Training configuration (see DEFAULT_CONFIG).
    device : str
        Torch device: "auto", "cpu", "cuda", "mps".

    Returns
    -------
    results : dict
        Summary statistics for this condition/seed:
          - final_mean_reward   : mean eval reward at end of training
          - final_std_reward    : std  eval reward at end of training
          - transfer_mean       : mean eval reward on test distribution
          - transfer_std        : std  eval reward on test distribution
          - training_time_s     : wall-clock training time
          - total_timesteps     : total env steps taken
    """
    label = f"{mode}_seed{seed}"
    print("\n" + "═" * 60)
    print(f"  Training: mode={mode}  seed={seed}")
    print("═" * 60)

    # ── Directories ───────────────────────────────────────────────────────────
    log_dir = Path(cfg["log_dir"]) / "logs" / label
    ckpt_dir = Path(cfg["log_dir"]) / "checkpoints" / label
    res_dir = Path(cfg["log_dir"]) / "results" / label
    for d in (log_dir, ckpt_dir, res_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Environments ──────────────────────────────────────────────────────────
    train_env = build_training_env(cfg, mode, seed)
    eval_env = build_eval_env(cfg, mode, seed, vec_normalize=train_env)

    # ── SAC model ─────────────────────────────────────────────────────────────
    model = SAC(
        env=train_env,
        seed=seed,
        device=device,
        tensorboard_log=str(log_dir),
        **SAC_HYPERPARAMS,
    )
    print(f"  Model parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = CallbackList(
        [
            CDRLoggingCallback(),
            TimingCallback(log_every=1000),
            LiveRenderCallback(render_freq=200, slowdown=0.0),
            CheckpointCallback(
                save_freq=cfg["checkpoint_freq"],
                save_path=str(ckpt_dir),
                name_prefix=f"auv_{label}",
                save_vecnormalize=True,
                verbose=0,
            ),
            EvalCallback(
                eval_env=eval_env,
                best_model_save_path=str(log_dir),
                log_path=str(log_dir),
                eval_freq=cfg["eval_freq"],
                n_eval_episodes=cfg["n_eval_episodes"],
                deterministic=True,
                render=False,
                verbose=1,
            ),
        ]
    )

    # ── Training ──────────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=callbacks,
        tb_log_name=label,
        reset_num_timesteps=True,
        progress_bar=True,
    )
    training_time = time.perf_counter() - t_start
    print(f"\n  Training complete in {training_time / 60:.1f} min")

    # ── Save final model ──────────────────────────────────────────────────────
    final_model_path = str(res_dir / "final_model")
    model.save(final_model_path)
    train_env.save(str(res_dir / "vecnormalize.pkl"))
    print(f"  Final model saved → {final_model_path}.zip")

    # ── Save CDR state (curriculum mode only) ─────────────────────────────────
    if mode == "curriculum":
        inner_env = train_env.venv.envs[0]
        if isinstance(inner_env, AUVDomainRandomWrapper):
            cdr_state = inner_env.get_cdr_state()
            cdr_path = res_dir / "cdr_state.json"
            with open(cdr_path, "w") as f:
                json.dump(cdr_state, f, indent=2)
            print(f"  CDR state saved → {cdr_path}")

    # ── Evaluation on training distribution ───────────────────────────────────
    # Sync normalisation stats before evaluation
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms

    mean_r, std_r = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=cfg["n_eval_episodes"],
        deterministic=True,
    )
    print(f"  Final eval (training dist): {mean_r:.2f} ± {std_r:.2f}")

    # ── Transfer evaluation on held-out test distribution ─────────────────────
    test_env = build_eval_env(
        cfg,
        mode=mode,
        seed=seed,
        vec_normalize=train_env,
        test_distribution=True,
    )
    transfer_mean, transfer_std = evaluate_policy(
        model,
        test_env,
        n_eval_episodes=cfg["n_eval_episodes"],
        deterministic=True,
    )
    test_env.close()
    print(f"  Transfer eval (test dist):  {transfer_mean:.2f} ± {transfer_std:.2f}")

    # ── Save eval results ─────────────────────────────────────────────────────
    np.savez(
        str(res_dir / "eval_transfer.npz"),
        mean=transfer_mean,
        std=transfer_std,
    )

    # ── Save training config ──────────────────────────────────────────────────
    config_snapshot = {
        "mode": mode,
        "seed": seed,
        "total_timesteps": cfg["total_timesteps"],
        "sac_hyperparams": {k: str(v) for k, v in SAC_HYPERPARAMS.items()},
        "training_time_s": training_time,
        "final_mean_reward": mean_r,
        "transfer_mean": transfer_mean,
    }
    with open(res_dir / "training_config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)

    train_env.close()
    eval_env.close()

    return {
        "mode": mode,
        "seed": seed,
        "final_mean_reward": mean_r,
        "final_std_reward": std_r,
        "transfer_mean": transfer_mean,
        "transfer_std": transfer_std,
        "training_time_s": training_time,
        "total_timesteps": cfg["total_timesteps"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Halcyon AUV with SAC under three DR conditions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        type=str,
        default="curriculum",
        choices=["none", "uniform", "curriculum", "all"],
        help=(
            "'none'=Naive SAC, 'uniform'=Uniform DR, "
            "'curriculum'=CDR (ours), 'all'=run all three sequentially."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Use multiple seeds for paper statistics.",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_CONFIG["total_timesteps"],
        help="Total training timesteps per condition.",
    )
    p.add_argument(
        "--xml-path",
        type=str,
        default=DEFAULT_CONFIG["xml_path"],
        help="Path to auv.xml MuJoCo model.",
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default=DEFAULT_CONFIG["log_dir"],
        help="Root directory for logs, checkpoints, and results.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="PyTorch device for SAC training.",
    )
    p.add_argument(
        "--eval-freq",
        type=int,
        default=DEFAULT_CONFIG["eval_freq"],
        help="Evaluate policy every N training steps.",
    )
    p.add_argument(
        "--n-eval-episodes",
        type=int,
        default=DEFAULT_CONFIG["n_eval_episodes"],
        help="Number of episodes per evaluation.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Validate XML path
    xml_path = Path(args.xml_path)
    if not xml_path.exists():
        # Try common locations
        candidates = [
            Path(__file__).parent / "envs" / "auv.xml",
            Path.home() / "rl_robotics" / "envs" / "auv.xml",
            Path("auv.xml"),
        ]
        xml_path = next((p for p in candidates if p.exists()), None)
        if xml_path is None:
            print("ERROR: auv.xml not found. Pass --xml-path explicitly.")
            sys.exit(1)

    cfg = dict(DEFAULT_CONFIG)
    cfg["total_timesteps"] = args.timesteps
    cfg["xml_path"] = str(xml_path)
    cfg["log_dir"] = args.log_dir
    cfg["eval_freq"] = args.eval_freq
    cfg["n_eval_episodes"] = args.n_eval_episodes

    # Determine which modes to run
    modes = ["none", "uniform", "curriculum"] if args.mode == "all" else [args.mode]

    # Run each condition and collect results
    all_results = []
    for mode in modes:
        result = train_condition(
            mode=mode,
            seed=args.seed,
            cfg=cfg,
            device=args.device,
        )
        all_results.append(result)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Training Summary")
    print("═" * 60)
    header = (
        f"{'Mode':<12} {'Seed':>4}  {'Train Reward':>14}  {'Transfer':>14}  {'Time':>8}"
    )
    print(header)
    print("-" * 60)
    for r in all_results:
        print(
            f"{r['mode']:<12} {r['seed']:>4}  "
            f"{r['final_mean_reward']:>7.2f} ± {r['final_std_reward']:<5.2f}  "
            f"{r['transfer_mean']:>7.2f} ± {r['transfer_std']:<5.2f}  "
            f"{r['training_time_s'] / 60:>6.1f}m"
        )

    # Save summary to log dir
    summary_path = Path(args.log_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Summary saved → {summary_path}")
    print("\n  Next: python plot_results.py --log-dir", args.log_dir)


if __name__ == "__main__":
    main()
