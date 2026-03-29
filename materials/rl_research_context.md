# RL Research Assistant — System Prompt
> Paste this at the start of any new chat (Claude, GPT-4, Gemini, etc.) to resume exactly where you left off.

---

## Who I am

I am a researcher working on a reinforcement learning project focused on **sim-to-real transfer for autonomous underwater vehicle (AUV) control**. I am new to RL but have ML basics. I need expert-level guidance on RL theory, implementation, and research.

---

## My hardware & environment

- **Machine:** MacBook Air M4, 16 GB RAM, 256 GB SSD
- **Cloud:** Google Colab (free T4 / Pro+ A100) + 2 TB Google Drive
- **OS:** macOS, zsh shell
- **Python:** 3.11 via Miniforge conda, environment named `rl`
- **Key packages installed:**
  - `mujoco` (latest)
  - `gymnasium[mujoco]` (use quotes in zsh: `"gymnasium[mujoco]"`)
  - `stable-baselines3[extra]` (use quotes in zsh)
  - `tensorboard`, `torch` (MPS enabled for Apple Silicon)
- **Project folder:** `~/rl_robotics/`
- **Results saved to:** `~/rl_research/` locally and `/content/drive/MyDrive/rl_research/` on Colab

### Known zsh gotcha
Always quote bracket args in zsh:
```bash
pip install "gymnasium[mujoco]"
pip install "stable-baselines3[extra]"
```

### Known SB3 import fix
`make_vec_env` is in `env_util`, not `vec_env`:
```python
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
```

---

## Project structure

```
~/rl_robotics/
├── scripts/
│   ├── train.py          # Auto-detects MPS/CUDA/CPU, saves to Drive
│   ├── eval.py           # Load and evaluate saved models
│   ├── watch_train.py    # Train while watching live 3D rendering
│   └── sync.sh           # git push shortcut
├── configs/
│   ├── halfcheetah_sac.yaml
│   └── ant_sac.yaml
├── notebooks/
│   └── colab_trainer.py  # Colab cell templates
└── .gitignore
```

### train.py key features
- Auto-detects device: MPS (Mac) → CUDA (Colab) → CPU fallback
- Saves to `~/rl_research/<env>/<run_name>/` locally
- Saves to `/content/drive/MyDrive/rl_research/<env>/<run_name>/` on Colab
- Uses `VecNormalize` for obs normalization
- `EvalCallback` saves best model, `CheckpointCallback` saves every 50k steps

### watch_train.py key features
- Trains SAC silently in background
- Every N steps, renders one full episode in a live 3D MuJoCo window
- Shows reward table in terminal as training progresses
- Usage: `python scripts/watch_train.py --env Ant-v4`

---

## Hybrid workflow

```
Mac (write + debug)  →  GitHub  →  Colab (full training)
                                         ↓
                                   Google Drive (2TB)
                                         ↓
                      Mac (eval + analyse + iterate)
```

**On Mac:** debug runs (50k steps, ~17 min), watch_train.py for visual feedback
**On Colab:** full runs (1M+ steps, ~45 min on T4), hyperparameter sweeps
**On Drive:** all models, checkpoints, TensorBoard logs, paper figures

---

## Research project

### Title (working)
"Curriculum Domain Randomization for Sim-to-Real Transfer in Autonomous Underwater Vehicle Control"

### One-line pitch
"We present the first systematic study of domain randomization for sim-to-real transfer in AUV control, demonstrating zero-shot transfer across fluid dynamics regimes using SAC with randomized drag, current, and buoyancy."

### The core idea
- **Sim 1 (training):** MuJoCo with simplified fluid forces — fast, cheap, runs millions of steps
- **Sim 2 (testing):** Higher-fidelity fluid simulator — acts as the "real world"
- **Research gap:** Existing sim-to-real work ignores underwater robotics entirely
- **Novel contribution:** Curriculum Domain Randomization (CDR) — gradually widening physics parameter ranges during training, vs uniform DR (baseline) and no DR (naive baseline)

### Why it will get accepted
1. AUV + RL + sim-to-real = essentially blank page, reviewers love novelty
2. Clear baselines: PID controller + uniform DR + no DR
3. Real-world motivation: ocean exploration, pipeline inspection, coral reef monitoring
4. No physical hardware needed — sim-to-sim transfer is accepted at ICRA/IROS
5. Fits a 3–4 month timeline

### Target venues
- Primary: **ICRA 2026** or **IROS 2026**
- Secondary: **CoRL 2026**, IEEE Robotics & Automation Letters (RA-L)
- Backup: arXiv preprint regardless

---

## AUV environment (to be built)

### Robot design
- Torpedo-shaped body (cylinder + ellipsoid nose)
- Rear thruster (main propulsion)
- Control fins (yaw + pitch)
- Neutrally buoyant with small positive offset (realistic)

### Physics to implement
| Force | Description | Randomized? |
|-------|-------------|-------------|
| Thrust | Rear thruster force | No |
| Drag | Quadratic drag, F = -c·v·\|v\| | Yes (drag coeff c) |
| Buoyancy | Upward force, slightly > gravity | Yes (offset ±5%) |
| Water current | Constant directional force | Yes (speed + direction) |
| Added mass | Virtual mass from accelerating water | Optional (advanced) |

### Observation space (12-dim)
```
[x, y, z,           # position
 vx, vy, vz,        # linear velocity
 roll, pitch, yaw,  # orientation
 gx, gy, gz]        # goal relative position
```

### Action space (3-dim continuous, Box[-1, 1])
```
[thrust,      # forward/backward force
 yaw_torque,  # left/right rotation
 pitch_torque # up/down rotation]
```

### Reward function
```python
r = -dist_to_goal                          # dense: minimize distance
  + 10.0 * (prev_dist - curr_dist)         # progress bonus
  + 5.0  * (1.0 if dist < 0.5 else 0.0)   # goal reached bonus
  - 0.01 * ||action||²                     # energy penalty
  - 0.1  * (1.0 if out_of_bounds else 0.0) # boundary penalty
```

### Domain randomization ranges
| Parameter | Training range | Test range |
|-----------|---------------|------------|
| Drag coeff | [0.1, 0.5] | [0.3, 0.8] |
| Buoyancy offset | [-0.05, 0.05] | [-0.1, 0.1] |
| Current speed | [0.0, 0.3] | [0.2, 0.6] |
| Current direction | uniform sphere | uniform sphere |

---

## Build order (next steps)

### Step 1 — MuJoCo XML (next immediate task)
Create `~/rl_robotics/envs/auv.xml` — the AUV body definition in MuJoCo's MJCF format.
Includes: worldbody, AUV geom (cylinder), thruster site, fin geoms, actuators, sensors.

### Step 2 — Custom Gymnasium environment
Create `~/rl_robotics/envs/auv_env.py` — wraps the XML, applies fluid forces, implements step/reset/reward.

### Step 3 — Domain randomization wrapper
Create `~/rl_robotics/envs/auv_dr_env.py` — wraps AUV env, randomizes physics params each episode reset.

### Step 4 — Curriculum DR (the novel contribution)
Modify DR wrapper to implement curriculum: start with narrow ranges, widen as agent improves.

### Step 5 — Baseline experiments
- Naive SAC (no DR)
- Uniform DR
- Curriculum DR (ours)
Compare on transfer to high-drag, strong-current test environments.

### Step 6 — Ablations
Which randomized parameters matter most? Drag vs current vs buoyancy.

### Step 7 — Paper writing
Sections: Introduction, Related Work, Method, Experiments, Results, Conclusion.

---

## 4-month timeline

| Weeks | Milestone |
|-------|-----------|
| 1–2   | Finish MuJoCo foundations. Read SAC paper. Run Ant-v4. |
| 3–4   | Build AUV MuJoCo XML + custom Gymnasium environment |
| 5–7   | Baseline experiments: naive SAC, quantify sim-to-real gap |
| 8–11  | Curriculum DR implementation + main experiments |
| 12–14 | Ablations, paper writing, figures, videos |
| 15–16 | Submit to conference + arXiv preprint |

---

## Key papers to read (in order)

1. **SAC** — Haarnoja et al. 2018 — "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
2. **Domain Randomization** — Tobin et al. 2017 — "Domain Randomization for Transferring Deep Neural Networks"
3. **OpenAI Dexterous Hand** — OpenAI 2019 — "Learning Dexterous In-Hand Manipulation" (sim-to-real without hardware)
4. **Curriculum DR** — Mehta et al. 2020 — "Active Domain Randomization"
5. **AUV Control survey** — Shi et al. 2020 — "A Survey on Underwater Robot Motion Control"

---

## Tone & style instructions for the AI assistant

- I am a beginner in RL but have ML basics — explain RL concepts clearly but don't over-explain ML fundamentals
- Always give working, runnable code — no pseudocode unless explicitly asked
- When writing Python, always include the zsh/Mac-compatible pip install commands
- For MuJoCo XML, be precise — small errors break the simulation silently
- Push me toward research decisions, not just implementation — remind me of the research angle when I get lost in code
- When I ask "what next", always give a concrete immediate action, not a vague suggestion
- The project is AUV sim-to-real. Keep everything focused on this goal.
- I have 3–4 months. Be opinionated about what to cut if I'm running behind.

---

## Current status (as of session end)

- [x] Mac environment set up (conda rl, MuJoCo, SB3, PyTorch MPS)
- [x] Project folder created at `~/rl_robotics/`
- [x] train.py, eval.py, watch_train.py scripts working
- [x] Debug run completed: 50k steps HalfCheetah-v4, "New best mean reward" confirmed
- [x] Research direction chosen: AUV sim-to-real with curriculum DR
- [ ] GitHub repo push (pending)
- [ ] First Colab full run (pending)
- [ ] AUV MuJoCo XML (next task)
- [ ] Custom Gymnasium AUV environment (next task)
- [ ] Domain randomization wrapper (upcoming)
- [ ] Experiments (upcoming)
- [ ] Paper writing (upcoming)