# Halcyon-v0

**Curriculum Domain Randomisation for Robust Sim-to-Real Transfer in Autonomous Underwater Vehicle Control**

> **Author:** [Limon Howlader] · EEE Undergraduate · MIST, Bangladesh  
> **Status:** Experiments running · Paper in preparation · Submitting to IEEE RA-L 2026  
> **Paper:** [arXiv link — will be added on submission day]  
> **GitHub:** https://github.com/Itzlimon22/rl_robotics.git

---

## What this is

This repository contains the complete research pipeline for an independent study on **curriculum domain randomisation (CDR)** for autonomous underwater vehicle (AUV) sim-to-real transfer.

The central question: does gradually expanding physics randomisation during training produce policies that transfer better to unseen fluid dynamics regimes than randomising from the full range from day one?

Three conditions are compared using SAC trained for 1 million steps each, evaluated across 3 random seeds, with zero-shot transfer to held-out physics parameters.

---

## The core idea

Standard domain randomisation throws the full range of physics at the agent from episode one — like throwing a beginner swimmer into a storm. Many agents fail to learn anything useful.

Curriculum DR starts with gentle physics and gradually makes it harder as the agent improves:
```
Episode reset
    ↓
Sample physics from current ranges
    ↓
Run episode (500 steps · 20 seconds simulated)
    ↓
Update rolling success window (W = 50 episodes)
    ↓
Success > 70%  →  expand ranges by 5%
Success < 40%  →  contract ranges by 3%
    ↓
curriculum_level ∈ [0, 1] tracks expansion progress
```

---

## Three experimental conditions

| Condition | Code | Description |
|-----------|------|-------------|
| No DR | `mode="none"` | Fixed nominal physics, no randomisation |
| Uniform DR | `mode="uniform"` | Full randomisation range from episode 1 |
| Curriculum DR | `mode="curriculum"` | Performance-gated range expansion (ours) |

---

## The Halcyon X4 AUV

Custom MuJoCo model of a torpedo-shaped autonomous underwater vehicle.

**Physical specifications:**
- Length: 1.0m · Diameter: 0.15m
- 4-thruster X-configuration at rear
- Full 6-DOF control (surge, sway, heave, roll, pitch, yaw)
- Maximum thrust: 20N per thruster

**Sensor suite (27 floats):**
- Position (3) · Quaternion (4) · Linear velocity (3)
- Angular velocity (3) · Accelerometer (3) · Gyroscope (3)
- Actuator forces (4) · Rangefinder (4)

**Observation space (18-dim, body frame):**

| Index | Description |
|-------|-------------|
| [0:3] | Goal direction vector in body frame (unit vector) |
| [3] | Distance to goal (m, clipped to 20) |
| [4:7] | Linear velocity in body frame (m/s) |
| [7:10] | Angular velocity in body frame (rad/s) |
| [10:13] | Euler angles — roll, pitch, yaw (rad) |
| [13:16] | Previous action |
| [16] | Water current speed (m/s) |
| [17] | Depth (m) |

**Action space (4-dim continuous):**
Four thruster commands ∈ [-1, 1]. Gear ratio 20 → 20N max per thruster.

**Fluid physics (applied every substep via xfrc_applied):**
1. Quadratic drag — axial and lateral independently in body frame
2. Buoyancy — net upward force with configurable offset
3. Water current drag — relative velocity drag from current
4. Added mass — virtual inertia from accelerating water

---

## Physics parameters randomised

| Parameter | CDR start range | Full range (UDR) | Held-out test range |
|-----------|----------------|------------------|---------------------|
| Lateral drag coefficient | [0.15, 0.25] | [0.10, 0.50] | [0.30, 0.80] |
| Axial drag coefficient | [0.06, 0.10] | [0.04, 0.20] | [0.12, 0.32] |
| Buoyancy offset | [-0.01, 0.05] | [-0.05, 0.10] | [-0.10, 0.15] |
| Current speed (m/s) | [0.0, 0.10] | [0.0, 0.30] | [0.20, 0.60] |
| Added mass coefficient | [0.10, 0.20] | [0.05, 0.30] | [0.20, 0.50] |
| Actuator efficiency | [0.95, 1.00] | [0.80, 1.00] | [0.70, 1.00] |

---

## Algorithm

**SAC (Soft Actor-Critic)** — Haarnoja et al. 2018

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 3e-4 |
| Replay buffer | 500,000 transitions |
| Batch size | 256 |
| Discount factor γ | 0.99 |
| Network architecture | MLP [256, 256], ReLU |
| Entropy coefficient | Auto-tuned |
| Training steps | 1,000,000 per condition |
| Frame skip | 4 (25Hz control rate) |
| Episode length | 500 steps (20 seconds) |

**VecNormalize** applied to observations and rewards (clip=10).

---

## Reward function

| Component | Weight | Description |
|-----------|--------|-------------|
| Progress | ×10.0 | Reward for closing distance to goal each step |
| Goal bonus | +50.0 | Terminal bonus on reaching goal |
| Alive | ×0.1 | Small per-step survival bonus |
| Energy | ×0.02 | Penalty for thruster effort (ctrl²) |
| Smoothness | ×0.05 | Penalty for action jerk |
| Orientation | ×0.5 | Penalty for not facing goal |
| Boundary | ×5.0 | Penalty for leaving workspace |

---

## Results

*Results will be updated as training runs complete. All conditions trained for 1M steps × 3 seeds.*

### Training distribution (Cell 7 evaluation)

| Condition | Seed 0 | Seed 1 | Seed 2 | Mean ± Std |
|-----------|--------|--------|--------|------------|
| None | - | - | - | - |
| Uniform DR | - | - | - | - |
| Curriculum DR | - | - | - | - |

### Held-out test distribution (eval.py)

| Condition | Seed 0 | Seed 1 | Seed 2 | Mean ± Std |
|-----------|--------|--------|--------|------------|
| None | - | - | - | - |
| Uniform DR | - | - | - | - |
| Curriculum DR | - | - | - | - |

---

## Repository structure
```
rl_robotics/
├── envs/
│   ├── auv.xml              — MuJoCo model (Halcyon X4 AUV)
│   ├── auv_env.py           — Gymnasium environment with fluid physics
│   └── auv_dr_wrapper.py    — DR wrapper (none / uniform / curriculum)
├── scripts/
│   ├── train.py             — SAC training pipeline
│   └── eval.py              — Held-out evaluation (coming soon)
├── notebooks/
│   └── colab_auv_training.ipynb  — Google Colab training notebook
├── paper/
│   └── RESEARCH_PLAN.md     — Full paper plan
├── results/
│   └── .gitkeep
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation
```bash
# Clone
git clone https://github.com/Itzlimon22/Obhyash-complete-project
cd Obhyash-complete-project

# Create environment
conda create -n rl python=3.11
conda activate rl

# Install dependencies
pip install -r requirements.txt
```

---

## How to run

### Quick test (local Mac, ~2 minutes)
```bash
conda activate rl
python scripts/train.py --mode curriculum --seed 0 --steps 500 --run-name test
```

### Full training run (local, ~8 hours)
```bash
python scripts/train.py --mode curriculum --seed 0 --steps 1000000
python scripts/train.py --mode uniform --seed 0 --steps 1000000
python scripts/train.py --mode none --seed 0 --steps 1000000
```

### Google Colab (recommended, ~4.5 hours on T4 GPU)
1. Upload `notebooks/colab_auv_training.ipynb` to Google Colab
2. Mount Google Drive
3. Set `MODE`, `SEED`, `STEPS` in the config cell
4. Run all cells

### TensorBoard
```bash
# All runs
tensorboard --logdir ~/rl_research/auv/

# Single run
tensorboard --logdir ~/rl_research/auv/curriculum/curriculum_seed0/tensorboard
```

---

## Key design decisions

**Why body-frame observations?**
Body-frame observations are invariant to world position and heading. This is essential for generalisation — a policy that works at position (0,0,0) should also work at (5,3,-2) without retraining.

**Why SAC over PPO?**
SAC is off-policy and significantly more sample efficient for continuous control tasks. Critical when each training run costs 4-5 hours of GPU time.

**Why curriculum DR over uniform DR?**
Uniform DR exposes the agent to maximum physics difficulty from episode one. Early training is dominated by catastrophic failures rather than useful learning signals. CDR stabilises early training by starting with manageable physics and expanding as the agent improves.

**Why frame_skip=4?**
25Hz control rate matches realistic AUV thruster update rates and reduces training time 4× versus frame_skip=1.

---

## Limitations

- No hydrodynamic coupling between axes (simplified drag model)
- No thruster dynamics (instantaneous thrust application)
- No real hardware validation — cross-simulator transfer to UUV Simulator used as proxy
- Sim-to-sim transfer only — real ocean deployment not tested

---

## Roadmap

- [x] MuJoCo AUV environment
- [x] SAC training pipeline
- [x] CDR wrapper (none / uniform / curriculum)
- [x] Google Colab notebook
- [ ] eval.py — held-out test distribution evaluation
- [ ] 9 main training runs (3 conditions × 3 seeds)
- [ ] 15 ablation runs (5 parameters × 3 seeds)
- [ ] PID baseline
- [ ] UUV Simulator zero-shot transfer
- [ ] Paper submission to IEEE RA-L

---

## Paper

**Title:** Curriculum Domain Randomisation for Robust Sim-to-Real Transfer in Autonomous Underwater Vehicle Control

**Target venue:** IEEE Robotics and Automation Letters (RA-L)

**arXiv:** [link — will be added on submission day]

**Abstract:** We present the first systematic study of curriculum domain randomisation (CDR) for sim-to-real transfer in autonomous underwater vehicle control. Policies trained with standard uniform domain randomisation struggle in early training due to high-variance fluid dynamics. Our CDR approach gradually expands randomisation ranges as the agent improves, stabilising training and achieving broader parameter coverage. We evaluate three conditions across 3 seeds in MuJoCo and perform zero-shot transfer to UUV Simulator on held-out fluid dynamics regimes.

---

## Citation
```bibtex
@article{[Limon Howlader]2026auvcdr,
  title   = {Curriculum Domain Randomisation for Robust Sim-to-Real
             Transfer in Autonomous Underwater Vehicle Control},
  author  = {[Limon Howlader]},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026},
  note    = {Under review}
}
```

---

## Contact

**[Limon Howlader]**  
B.Sc. Electrical, Electronics and Communication Engineering  
Military Institute of Science and Technology (MIST), Dhaka, Bangladesh  
[limon.eece22@gmail.com]  
[LinkedIn: linkedin.com/in/yourname]  
[arXiv: link when available]

---

*This is independent research conducted outside of any formal lab or institution.*  
*Started: 2025 · Expected submission: mid 2026*