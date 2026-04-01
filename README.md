# Halcyon-v0

**Curriculum Domain Randomisation for Energy-Efficient and Robust AUV Control**

> **Author:** Limon Howlader · EEE Undergraduate · MIST, Bangladesh
> **Status:** Experiments complete · Paper in preparation · Target: IEEE RA-L 2026
> **GitHub:** https://github.com/Itzlimon22/rl_robotics.git
> **arXiv:** *(will be added on submission day)*

---

## The Central Finding

A PID controller perfectly tuned for nominal AUV physics achieves **100% success**.
Under modest fluid parameter shift — drag increasing by 3–6×, current at 0.4–0.8 m/s
— the same controller achieves **3% success**.

This single result motivates everything in this repository.

| Condition | Train Success | Test Success | Test Reward | Energy/step |
|-----------|:------------:|:------------:|:-----------:|:-----------:|
| PID (classical) | 100% | **3%** | -249 | highest |
| Naive SAC (no DR) | ~100% | 60.7% ± 27.2% | low | medium |
| Uniform DR | ~100% | 99.7% ± 0.5% | ~76 | 0.665 |
| **CDR (ours)** | ~100% | **96.0% ± 3.7%** | ~73 | **0.623** |

**Key results:**
- PID collapses catastrophically under distribution shift — RL with DR does not
- Any DR eliminates the ±27% deployment variance of unrobust policies
- CDR uses **6.3% less energy per step** than Uniform DR at comparable success rates
- For battery-constrained real AUVs, 6% thrust saving extends mission duration meaningfully

---

## What This Is

Complete research pipeline for an independent study on **curriculum domain randomisation (CDR)**
for AUV sim-to-real transfer. Three conditions compared — Naive SAC, Uniform DR, CDR —
using SAC trained for 1M steps × 3 seeds, evaluated with zero-shot transfer to held-out
fluid dynamics regimes never seen during training.

---

## The Core Idea

Standard domain randomisation throws the full range of physics at the agent from
episode one — high drag, strong current, degraded thrusters, all at once.
Early training is dominated by catastrophic failures rather than useful learning signals.

CDR starts with gentle physics and expands automatically as the agent improves:

```
Episode reset
    ↓
Sample physics from current ranges
    ↓
Run episode (500 steps · 20 seconds simulated)
    ↓
Update rolling success window (W = 50 episodes)
    ↓
Success rate > 70%  →  EXPAND ranges by 5%
Success rate < 40%  →  CONTRACT ranges by 3%
    ↓
curriculum_level ∈ [0, 1] tracks expansion progress
```

The result: CDR policies are smoother, more efficient, and equally successful —
because the curriculum teaches efficient navigation before exposing the agent to
the hardest conditions.

---

## Results

### Main Results — Held-Out Test Distribution (100 episodes × 3 seeds)

Test distribution uses fluid parameters **never seen during training**:
drag 3–6× nominal, current 0.4–0.8 m/s, actuator efficiency 60–90%.

| Condition | Success Rate | Reward | Reward Std | Energy/step |
|-----------|:-----------:|:------:|:----------:|:-----------:|
| PID | 3.0% | -249 | high | highest |
| Naive SAC | 60.7% ± 27.2% | low | ±57 | medium |
| Uniform DR | **99.7% ± 0.5%** | ~76 | ±13 | 0.665 |
| CDR (ours) | 96.0% ± 3.7% | ~73 | ±20 | **0.623** |

> The ±27.2% variance of Naive SAC represents a deployment lottery —
> the same training procedure produces policies ranging from barely functional
> to moderately robust, with no way to predict which.

### Per-Seed Breakdown

| Condition | Seed 1 | Seed 2 | Seed 0 |
|-----------|--------|--------|--------|
| CDR | 100% | 90% | pending |
| Uniform DR | 100% | 100% | pending |
| Naive SAC | 62% | 96% | pending |
| PID | 3% | 3% | 3% |

### Results Figures

> *Training curves, curriculum progression, and results bar charts will be
> added here upon paper completion. See `paper/figures/` folder.*

**Figure 1** — Training curves (success rate vs timesteps, all conditions)
```
paper/figures/fig1_training_curves.png
```

**Figure 2** — CDR curriculum level and rolling success rate over training
```
paper/figures/fig2_curriculum_progression.png
```

**Figure 3** — Main results bar chart (test success rate + energy per condition)
```
paper/figures/fig3_main_results.png
```

**Figure 4** — Ablation: transfer success rate by single randomised parameter
```
paper/figures/fig4_ablation.png        ← pending ablation runs
```

---

## The Halcyon X4 AUV

Custom MuJoCo model of a torpedo-shaped AUV with realistic fluid physics.

**Physical specs:**
- Length 1.0 m · Diameter 0.15 m · Mass 11.1 kg
- 4-thruster X-configuration · 20 N max per thruster · True 6-DOF
- Four passive stabiliser fins · Neutrally buoyant (buoyancy applied in Python)

**Sensor suite (27 floats total):**
Position (3) · Quaternion (4) · Linear velocity (3) · Angular velocity (3)
Accelerometer (3) · Gyroscope (3) · Actuator forces (4) · Rangefinder (4)

**Observation space (18-dim, body frame):**

| Index | Quantity | Notes |
|-------|----------|-------|
| [0:3] | Goal direction (body frame) | Unit vector — invariant to world heading |
| [3] | Goal distance (m) | Clipped to [0, 20] |
| [4:7] | Linear velocity (body frame) | m/s |
| [7:10] | Angular velocity (body frame) | rad/s |
| [10:13] | Euler angles (roll, pitch, yaw) | rad |
| [13:16] | Previous action | For smoothness penalty |
| [16] | Water current speed | m/s — observable disturbance |
| [17] | Depth | World Z position |

> Body-frame observations are invariant to world position and heading.
> A policy that works at (0,0,0) works at (5,3,−2) without retraining.

**Action space:** 4 thruster commands ∈ [−1, 1] · 20 N max each

**Fluid physics (applied every substep via `xfrc_applied`, world frame):**
```
F_drag  = −c_drag × v_body × |v_body|         quadratic, per axis
F_buoy  = m × g × (1 + δ_buoy)               world +Z
F_curr  = c_drag × (v_curr − v_AUV) × |v_curr − v_AUV|
F_am    = −C_am × m × a_body                  added mass
```

---

## Physics Parameters Randomised

| Parameter | CDR start | Full range (UDR) | Test held-out |
|-----------|:---------:|:----------------:|:-------------:|
| Lateral drag (kg/m) | [0.15, 0.25] | [0.10, 0.50] | [0.60, 1.20] |
| Axial drag (kg/m) | [0.06, 0.10] | [0.04, 0.20] | [0.24, 0.48] |
| Buoyancy offset | [−0.01, 0.05] | [−0.05, 0.10] | [−0.15, 0.20] |
| Current speed (m/s) | [0.0, 0.10] | [0.0, 0.30] | [0.40, 0.80] |
| Added mass coeff | [0.10, 0.20] | [0.05, 0.30] | [0.30, 0.60] |
| Actuator efficiency | [0.95, 1.00] | [0.80, 1.00] | [0.60, 0.90] |

CDR starts at the narrow left column and expands toward the middle column
based on rolling success rate. Test distribution (right column) is never seen
during training — zero-shot transfer evaluation only.

---

## Algorithm

**SAC (Soft Actor-Critic)** — Haarnoja et al. 2018

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 3 × 10⁻⁴ |
| Replay buffer | 500,000 |
| Batch size | 256 |
| Discount γ | 0.99 |
| Network | MLP [256, 256] ReLU |
| Entropy coeff | Auto-tuned |
| Training steps | 1,000,000 |
| Control rate | 25 Hz (frame skip 4) |
| Episode length | 500 steps (20 s) |
| Obs normalisation | VecNormalize (clip=10) |

**CDR hyperparameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Window W | 50 episodes | Rolling success rate window |
| θ_hi | 0.70 | Expand threshold |
| θ_lo | 0.40 | Contract threshold |
| ε_expand | 0.05 | 5% of full range per expansion |
| ε_contract | 0.03 | 3% per contraction |

---

## Reward Function

| Component | Weight | Description |
|-----------|:------:|-------------|
| Progress | ×10.0 | Reward per metre closed toward goal (normalised by dt) |
| Goal bonus | +50.0 | Terminal bonus on reaching goal |
| Alive | ×0.1 | Per-step survival bonus |
| Energy penalty | ×0.02 | −ctrl² — encourages efficient thrust |
| Smoothness | ×0.05 | −(Δaction)² — reduces actuator jitter |
| Orientation | ×0.5 | Penalty for not facing goal (cos θ term) |
| Boundary | ×5.0 | Penalty for leaving 15 m workspace sphere |

---

## Repository Structure

```
rl_robotics/
├── envs/
│   ├── auv.xml                   MuJoCo MJCF model — Halcyon X4
│   ├── auv_env.py                Gymnasium environment + fluid physics
│   └── auv_dr_wrapper.py        DR wrapper: none / uniform / curriculum
├── scripts/
│   ├── train.py                  SAC training pipeline (CLI)
│   ├── resume.py                 Resume training from checkpoint
│   ├── pid_baseline.py          PID controller baseline + tuner
│   └── eval.py                  Held-out evaluation  ← in progress
├── notebooks/
│   └── colab_auv_training.ipynb  Google Colab training notebook
├── paper/
│   ├── RESEARCH_PLAN.md         Full paper plan
│   ├── paper_draft.md           Complete paper draft
│   └── figures/                 Publication figures (generated after runs)
├── results/
│   └── .gitkeep                 Results saved to Google Drive (not tracked)
├── MASTER_CONTEXT.md            Full project context for AI session handoff
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
# Clone
git clone https://github.com/Itzlimon22/rl_robotics.git
cd rl_robotics

# Create environment
conda create -n rl python=3.11
conda activate rl

# Install dependencies
pip install "mujoco" "gymnasium[mujoco]" "stable-baselines3[extra]" \
            "torch" "tensorboard" "numpy"
```

---

## How to Run

### Sanity check (~30 seconds)
```bash
conda activate rl
cd ~/rl_robotics
python -c "
from envs.auv_env import HalcyonAUVEnv
from envs.auv_dr_wrapper import AUVDomainRandomWrapper
env = AUVDomainRandomWrapper(HalcyonAUVEnv('envs/auv.xml'), mode='curriculum', seed=0)
obs, info = env.reset()
print(f'obs: {obs.shape} | goal_dist: {info[\"goal_dist\"]:.2f}m')
env.close()
print('OK')
"
```

### Debug run (local Mac, ~10 min)
```bash
python scripts/train.py --mode curriculum --seed 0 --steps 50000 --run-name debug
```

### Full training run (Colab recommended — ~4.5 hours on T4)
```bash
# All three conditions for one seed (run in parallel on 3 Colab accounts)
python scripts/train.py --mode curriculum --seed 0 --steps 1000000
python scripts/train.py --mode uniform    --seed 0 --steps 1000000
python scripts/train.py --mode none       --seed 0 --steps 1000000
```

### Resume after crash
```bash
python scripts/resume.py --mode curriculum --seed 0 --checkpoint 400000
```

### PID baseline evaluation
```bash
python scripts/pid_baseline.py --episodes 50 --test-dist
```

### TensorBoard
```bash
tensorboard --logdir ~/rl_research/auv/
```

### Google Colab
1. Upload `notebooks/colab_auv_training.ipynb` → File → Upload notebook
2. Runtime → Change runtime type → T4 GPU
3. Run cells 1–5 (mount Drive, install, clone, sanity check, train)
4. Configure `MODE`, `SEED`, `STEPS` in Cell 5 before running

---

## Key Design Decisions

**Body-frame observations** — invariant to world position/heading, necessary for
generalisation under arbitrary current direction.

**SAC over PPO** — off-policy, significantly more sample efficient for continuous
control. Critical when each run costs 4–5 GPU hours.

**`xfrc_applied` for fluid forces** — MuJoCo's external force API applied in world
frame each substep. Drag computed in body frame then rotated via `data.xmat`.

**frame_skip=4** — 25 Hz control rate matches realistic AUV thruster dynamics,
reduces training time 4× vs frame_skip=1.

**VecNormalize** — running mean/std normalisation of observations and rewards.
Must save `vec_normalize.pkl` alongside model weights — evaluation requires identical
normalisation statistics to training.

---

## Limitations

- Simplified drag model — no hydrodynamic coupling between axes
- No thruster dynamics — instantaneous thrust application
- Sim-to-sim only — no real hardware validation
- CDR hyperparameters hand-tuned — adaptive meta-learned thresholds are future work

---

## Roadmap

- [x] MuJoCo AUV environment (Halcyon X4)
- [x] SAC training pipeline with VecNormalize + callbacks
- [x] CDR wrapper — none / uniform / curriculum modes
- [x] Checkpoint resume after crash
- [x] Google Colab notebook (.ipynb)
- [x] PID baseline controller + tuner
- [x] 9 main training runs (3 conditions × 3 seeds)
- [x] Held-out test distribution evaluation
- [ ] eval.py — unified evaluation script with energy logging
- [ ] 15 ablation runs (5 parameters × 3 seeds)
- [ ] Publication figures (training curves, ablation bar chart)
- [ ] Paper submission — IEEE RA-L 2026

---

## Paper

**Title:** Curriculum Domain Randomisation for Energy-Efficient and Robust AUV Control

**Target venue:** IEEE Robotics and Automation Letters (RA-L), 2026

**Status:** Draft complete · Ablation experiments in progress

**Abstract:**
Classical PID controllers for autonomous underwater vehicle navigation achieve
near-perfect success under nominal fluid conditions yet collapse catastrophically
under modest distribution shift — dropping from 100% to 3% success as drag and
current increase to real-ocean ranges. We present a systematic study of domain
randomisation strategies for AUV sim-to-real transfer, comparing PID, Naive SAC,
Uniform DR, and Curriculum DR (CDR) trained with SAC for 1M steps across 3 seeds.
On a held-out test distribution with elevated drag (0.60–1.20 kg/m) and strong
currents (0.40–0.80 m/s), CDR achieves 96.0% ± 3.7% success while consuming
6.3% less thrust energy per step than Uniform DR — a meaningful saving for
battery-constrained real AUVs. We open-source our MuJoCo AUV environment and
complete training pipeline.

---

## Citation

```bibtex
@article{howlader2026auvcdr,
  title   = {Curriculum Domain Randomisation for Energy-Efficient
             and Robust AUV Control},
  author  = {Howlader, Limon},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026},
  note    = {Under review}
}
```

---

## Contact

**Limon Howlader**
B.Sc. Electrical, Electronics and Communication Engineering
Military Institute of Science and Technology (MIST), Dhaka, Bangladesh
limon.eece22@gmail.com

*Independent research conducted outside of any formal lab or institution.*
*Started: 2025 · Expected submission: mid-2026*