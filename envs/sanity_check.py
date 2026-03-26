# sanity_check.py
# Run from: ~/rl_robotics/envs/
#   python sanity_check.py

import os
import sys
import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

sys.path.insert(0, THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

# ── Find auv.xml ───────────────────────────────────────────────────────────────
_CANDIDATES = [
    os.path.join(THIS_DIR, "auv.xml"),
    os.path.join(PROJECT_ROOT, "auv.xml"),
    os.path.expanduser("~/rl_robotics/envs/auv.xml"),
]

XML_PATH = next((p for p in _CANDIDATES if os.path.exists(p)), None)

if XML_PATH is None:
    print("ERROR: auv.xml not found. Searched in:")
    for p in _CANDIDATES:
        print(f"  {p}")
    sys.exit(1)

print(f"✓ Found auv.xml at: {XML_PATH}\n")

# ── Import ─────────────────────────────────────────────────────────────────────
from auv_dr_wrapper import make_auv_env

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Observation layout diagnostic
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1: OBSERVATION LAYOUT")
print("=" * 60)

env = make_auv_env(XML_PATH, mode="none", seed=0)
obs, info = env.reset()

print(f"Observation shape : {obs.shape}")
print(f"Action space      : {env.action_space}")
print()

print("── info dict ────────────────────────────────────────────")
for k, v in info.items():
    print(f"  {k:20s}: {v}")

print()
print("── Full observation vector ───────────────────────────────")
for i, val in enumerate(obs):
    print(f"  obs[{i:2d}] = {val:+.6f}")

# ── Try to match obs values to goal direction ──────────────────────────────────
if "goal_pos" in info and "auv_pos" in info:
    g = np.array(info["goal_pos"])
    p = np.array(info["auv_pos"])
    vec = g - p
    dist = np.linalg.norm(vec)
    direction = vec / (dist + 1e-8)

    print()
    print("── Goal direction (world frame, normalised) ──────────────")
    print(f"  X={direction[0]:+.4f}  Y={direction[1]:+.4f}  Z={direction[2]:+.4f}")
    print(f"  dist={dist:.4f} m")
    print()
    print("── Searching obs[] for matching values ───────────────────")
    found_any = False
    for i, val in enumerate(obs):
        for label, expected in [
            ("goal_dir_X", direction[0]),
            ("goal_dir_Y", direction[1]),
            ("goal_dir_Z", direction[2]),
            ("goal_dist", dist),
            ("-goal_dist", -dist),
        ]:
            if abs(float(val) - expected) < 0.05:
                print(f"  ✓ obs[{i:2d}] = {val:+.4f}  ≈  {label} ({expected:+.4f})")
                found_any = True
    if not found_any:
        print(
            "  ✗ No close matches found — goal may be in body frame, not world frame."
        )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Zero action (drift test)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("SECTION 2: ZERO ACTION (drift test)")
print("=" * 60)

obs, info = env.reset()
dist_start = info.get("goal_dist", float("nan"))
print(f"Start dist: {dist_start:.3f} m")

for step in range(5):
    obs2, reward, terminated, truncated, info2 = env.step(np.zeros(4, dtype=np.float32))
    print(
        f"  step {step + 1}: reward={reward:8.3f}  "
        f"goal_dist={info2.get('goal_dist', float('nan')):.3f} m"
    )
    if terminated or truncated:
        print("  Episode ended early.")
        break

print()
print("── Obs change after zero action (non-zero diffs only) ────")
for i, (v1, v2) in enumerate(zip(obs, obs2)):
    diff = float(v2) - float(v1)
    if abs(diff) > 1e-4:
        print(f"  obs[{i:2d}]: {v1:+.4f} → {v2:+.4f}  (Δ={diff:+.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Full forward thrust [1,1,0,0]
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("SECTION 3: FULL FORWARD THRUST [1, 1, 0, 0]")
print("=" * 60)

obs, info = env.reset()
print(f"Start dist: {info.get('goal_dist', '?'):.3f} m")

for step in range(10):
    obs, reward, terminated, truncated, info = env.step(
        np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    )
    print(
        f"  step {step + 1:2d}: reward={reward:8.3f}  "
        f"dist={info.get('goal_dist', float('nan')):.3f} m  "
        f"obs[0:6]={np.round(obs[0:6], 3)}"
    )
    if terminated or truncated:
        print("  Episode ended early.")
        break

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — All six pure actions
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("SECTION 4: SIX PURE ACTIONS (which one reduces distance?)")
print("=" * 60)

ACTIONS = {
    "forward   [+1,+1, 0, 0]": np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "backward  [-1,-1, 0, 0]": np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32),
    "up        [ 0, 0,+1,-1]": np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32),
    "down      [ 0, 0,-1,+1]": np.array([0.0, 0.0, -1.0, 1.0], dtype=np.float32),
    "roll+     [+1,-1, 0, 0]": np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32),
    "roll-     [-1,+1, 0, 0]": np.array([-1.0, 1.0, 0.0, 0.0], dtype=np.float32),
}

for label, action in ACTIONS.items():
    obs, info = env.reset(seed=42)  # same start every time
    dist_start = info.get("goal_dist", float("nan"))
    total_reward = 0.0

    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    dist_end = info.get("goal_dist", float("nan"))
    delta = dist_end - dist_start
    arrow = "✓ closer" if delta < -0.05 else ("✗ farther" if delta > 0.05 else "~ same")
    print(f"  {label}  |  Δdist={delta:+.2f} m  reward={total_reward:8.2f}  {arrow}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Reward breakdown
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("SECTION 5: REWARD BREAKDOWN (single step, goal direction)")
print("=" * 60)

obs, info = env.reset(seed=42)
dist_before = info.get("goal_dist", float("nan"))

# Try the action that moved closest in Section 4 — default to forward
action = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action)
dist_after = info.get("goal_dist", float("nan"))

print(f"  dist before : {dist_before:.4f} m")
print(f"  dist after  : {dist_after:.4f} m")
print(f"  Δ dist      : {dist_after - dist_before:+.4f} m")
print(f"  reward      : {reward:.4f}")
print()
print("  If |reward| >> |Δ dist|, an alive penalty is dominating.")
print("  Expected healthy reward ≈ -0.1 to -5.0 per step.")

env.close()

print()
print("=" * 60)
print("Paste the full output above — we will fix the indices and")
print("reward function based on what these sections reveal.")
print("=" * 60)
