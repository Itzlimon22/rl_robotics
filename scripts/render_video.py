"""
render_video.py — Render trained AUV policy as MP4 video
=========================================================
Loads a trained SAC model with correct VecNormalize statistics
and renders episodes to an MP4 file. Includes custom 2D Camera logic.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import gymnasium as gym
import mujoco

# ── Path setup ────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent.resolve()
_REPO_ROOT = _SCRIPT_DIR.parent
_ENVS_DIR = _REPO_ROOT / "envs"
for _p in [_REPO_ROOT, _ENVS_DIR, _SCRIPT_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── Check imageio ─────────────────────────────────────────────
try:
    import imageio
except ImportError:
    print("imageio not installed. Run:")
    print('  pip install "imageio[ffmpeg]"')
    sys.exit(1)

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from auv_env import HalcyonAUVEnv
from auv_dr_wrapper import AUVDomainRandomWrapper


# ─────────────────────────────────────────────────────────────
# Run directory finder
# ─────────────────────────────────────────────────────────────


def find_run_dir(mode: str, seed: int) -> Path:
    base = Path.home() / "rl_research" / "auv"

    # Try 1: exact match
    run_dir = base / mode / f"{mode}_seed{seed}"
    if run_dir.exists():
        print(f"[render] Found run dir: {run_dir}")
        return run_dir

    # Try 2: search all subfolders
    print(f"[render] Searching for mode='{mode}' seed={seed}...")
    candidates = []
    for folder in sorted(base.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        if mode in folder.name or folder.name in mode:
            for subfolder in sorted(folder.iterdir()):
                if not subfolder.is_dir():
                    continue
                if f"seed{seed}" in subfolder.name:
                    candidates.append(subfolder)

    if len(candidates) == 1:
        print(f"[render] Found: {candidates[0]}")
        return candidates[0]
    elif len(candidates) > 1:
        chosen = sorted(candidates)[-1]
        print(f"[render] Multiple matches, using: {chosen}")
        return chosen

    raise FileNotFoundError(f"Run directory not found for mode='{mode}' seed={seed}")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def get_dr_mode(mode: str) -> str:
    for condition in ["curriculum", "uniform", "none"]:
        if condition in mode:
            return condition
    return "curriculum"


def get_base_env(vec_env):
    env = vec_env.venv.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


# ─────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────


def make_render_env(mode: str, seed: int, xml_path: str):
    is_tracking = "tracking" in mode
    dr_mode = get_dr_mode(mode)

    if is_tracking:
        try:
            from auv_tracking_env import HalcyonAUVTrackingEnv

            base = HalcyonAUVTrackingEnv(
                xml_path=xml_path,
                path_speed=0.3,
                render_mode="rgb_array",
                render_width=640,
                render_height=480,
            )
        except ImportError:
            base = HalcyonAUVEnv(xml_path=xml_path, render_mode="rgb_array")
    else:
        base = HalcyonAUVEnv(xml_path=xml_path, render_mode="rgb_array")

    wrapped = AUVDomainRandomWrapper(base, mode=dr_mode, seed=seed + 999)
    wrapped.set_test_distribution()
    return wrapped


# ─────────────────────────────────────────────────────────────
# Model loader & Wrappers
# ─────────────────────────────────────────────────────────────


# FIXED: Now officially inherits from gym.Wrapper so SB3 accepts it
class RangefinderObsWrapper(gym.Wrapper):
    """Appends 4 rangefinder readings to match 23-dim obs space."""

    def __init__(self, env):
        super().__init__(env)
        base = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.append(base.low, np.full(4, -1.0, dtype=np.float32)),
            high=np.append(base.high, np.full(4, 20.0, dtype=np.float32)),
            dtype=np.float32,
        )

    def _rf(self, obs):
        try:
            # unwrapped allows us to access the deep mujoco data through other wrappers
            rf = self.unwrapped.data.sensordata[23:27].copy().astype(np.float32)
        except:
            rf = np.full(4, -1.0, dtype=np.float32)
        return np.append(obs, rf).astype(np.float32)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return self._rf(obs), info

    def step(self, a):
        obs, r, t, tr, info = self.env.step(a)
        return self._rf(obs), r, t, tr, info


def load_model_with_vecnorm(run_dir: Path, render_env):
    vn_path = run_dir / "vec_normalize.pkl"
    model_path = run_dir / "best_model.zip"

    if not model_path.exists():
        model_path = run_dir / "final_model.zip"

    tmp = SAC.load(str(model_path))
    model_obs_dim = tmp.observation_space.shape[0]
    env_obs_dim = render_env.observation_space.shape[0]

    if model_obs_dim != env_obs_dim:
        if model_obs_dim - env_obs_dim == 4:
            render_env = RangefinderObsWrapper(render_env)

    if vn_path.exists():
        vec_env = DummyVecEnv([lambda: render_env])
        vec_env = VecNormalize.load(str(vn_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        model = SAC.load(str(model_path), env=vec_env)
        return model, vec_env, True
    else:
        model = SAC.load(str(model_path))
        return model, None, False


# ─────────────────────────────────────────────────────────────
# Render loop & Camera Fix
# ─────────────────────────────────────────────────────────────


def apply_static_camera(env):
    """Forces the MuJoCo viewer into a perfect 2D top-down view."""
    core_env = env.unwrapped
    viewer = getattr(core_env, "mujoco_renderer", None)

    if viewer is not None:
        v = viewer._get_viewer(render_mode="rgb_array")
    else:
        v = getattr(core_env, "viewer", getattr(core_env, "_viewer", None))

    if v is not None:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.lookat[0] = 0.0
        v.cam.lookat[1] = 0.0
        v.cam.lookat[2] = 0.0
        v.cam.distance = 45.0  # Zoomed out to see all 5 obstacles
        v.cam.elevation = -90.0  # Perfect Top-Down View


def render_episodes(model, render_env, vec_env, n_episodes: int, use_vecnorm: bool):
    frames = []

    # Force initial render so the viewer object exists
    if use_vecnorm:
        get_base_env(vec_env).render()
    else:
        render_env.render()

    for ep in range(n_episodes):
        print(f"\n[render] --- Episode {ep + 1}/{n_episodes} ---")

        if use_vecnorm:
            obs = vec_env.reset()
            base_env = get_base_env(vec_env)
            done = False
            ep_n = 0

            while not done and ep_n < 1000:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done_arr, _ = vec_env.step(action)
                done = bool(done_arr[0])
                ep_n += 1

                apply_static_camera(base_env)  # Inject Camera Zoom
                frame = base_env.render()
                if frame is not None:
                    frames.append(frame)

        else:
            obs, _ = render_env.reset()
            done = False
            ep_n = 0

            while not done and ep_n < 1000:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = render_env.step(action)
                done = terminated or truncated
                ep_n += 1

                apply_static_camera(render_env)  # Inject Camera Zoom
                frame = render_env.render()
                if frame is not None:
                    frames.append(frame)

        print(f"  ✓ SUCCESS | frames={ep_n}")

    return frames


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────


def render_video(args):
    xml_path = str(_ENVS_DIR / "auv.xml")
    run_dir = find_run_dir(args.mode, args.seed)

    render_env = make_render_env(args.mode, args.seed, xml_path)
    model, vec_env, use_vn = load_model_with_vecnorm(run_dir, render_env)

    frames = render_episodes(model, render_env, vec_env, args.episodes, use_vn)

    out_path = args.out or f"auv_{args.mode}_seed{args.seed}.mp4"
    imageio.mimwrite(out_path, frames, fps=25, quality=8)
    print(f"\n✅ Video encoded to: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="master_curriculum", type=str)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    render_video(args)


if __name__ == "__main__":
    main()
