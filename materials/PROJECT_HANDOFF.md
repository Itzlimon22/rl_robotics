# AUV Research Project — Full Context Handoff
> **Purpose:** Paste this file at the start of any new AI session to resume exactly where we left off.
> **Project:** Curriculum Domain Randomisation for AUV Sim-to-Real Transfer (IROS 2026 paper)
> **Researcher:** Solo founder/researcher, based in Bangladesh, MacBook Air M4 + Google Colab

---

## 1. Who You Are Talking To

- Solo researcher building an RL research project for a **PhD scholarship application**
- Background: ML basics, new to RL, learning fast
- Hardware: MacBook Air M4, 16GB RAM + Google Colab (T4 GPU, 3 accounts)
- Python: 3.11, conda env named `rl`, project at `~/rl_robotics/`
- Results saved to: `~/rl_research/auv/` locally and `/content/drive/MyDrive/rl_research/auv/` on Colab

---

## 2. The Research Project

### Paper Title (working)
**"Curriculum Domain Randomisation for Robust Sim-to-Real Transfer in Autonomous Underwater Vehicle Control"**

### One-Line Pitch
First systematic study of curriculum domain randomisation for AUV sim-to-real transfer, demonstrating zero-shot transfer across fluid dynamics regimes using SAC with performance-gated parameter range expansion.

### Target Venue
- **Primary:** IROS 2026 (deadline: March 2, 2026) — main track
- **Stretch:** CoRL 2026 (deadline: May 28, 2026)
- **Backup:** IEEE RA-L (journal, anytime)
- **Always:** arXiv preprint on day of IROS submission

### Core Claim
CDR produces policies that transfer more robustly to unseen fluid dynamics regimes than Uniform DR (UDR), using the same training budget, because it stabilises early training and achieves better coverage of the task-relevant region of parameter space.

### Three Experimental Conditions (the paper comparison)
| Condition | Code | Paper Label |
|-----------|------|-------------|
| `mode="none"` | Naive SAC, no DR | Baseline 1 |
| `mode="uniform"` | Uniform DR, fixed ranges | Baseline 2 |
| `mode="curriculum"` | CDR, performance-gated expansion | **Ours** |

### Why This Is Novel
1. AUV + RL + sim-to-real = essentially blank page in literature
2. No prior work applies curriculum DR to underwater fluid physics
3. Two-fidelity evaluation: MuJoCo (train) → UUV Simulator (zero-shot transfer test)
4. Ablation study quantifying which fluid parameter dominates the sim-to-real gap

---

## 3. What Has Been Built (Complete)

### File Structure
```
~/rl_robotics/
├── envs/
│   ├── auv.xml              ✅ Done — Halcyon X4 MuJoCo model
│   ├── auv_env.py           ✅ Done — Gymnasium environment
│   └── auv_dr_wrapper.py    ✅ Done — DR wrapper (none/uniform/curriculum)
├── scripts/
│   └── train.py             ✅ Done — Full training pipeline
├── notebooks/
│   └── colab_auv_training.ipynb  ✅ Done — Colab notebook (proper .ipynb)
└── paper/
    └── RESEARCH_PLAN.md     ✅ Done — Full paper plan (554 lines)
```

### auv.xml — Halcyon X4 AUV (v2, fixed)
- Torpedo-shaped AUV, ~1m long, 0.15m diameter
- 4-thruster X-configuration at rear (true 6-DOF control)
- Gravity disabled — buoyancy applied manually in Python
- **27-float sensor suite:** pos(3), quat(4), linvel(3), angvel(3), accel(3), gyro(3), actuatorfrc(4), rangefinder(4)
- Keyframe "home" for clean reset
- **Key fix applied:** removed `<joint>` from `<default>` (was causing silent conflicts)

### auv_env.py — HalcyonAUVEnv
- Gymnasium-compatible, SB3 check_env passes
- **18-dim observation space** (body frame — invariant to world position/heading):
  - `[0:3]` goal_vec_body (unit vector), `[3]` goal_dist, `[4:7]` lin_vel_body
  - `[7:10]` ang_vel_body, `[10:13]` euler_angles, `[13:16]` prev_action
  - `[16]` current_speed, `[17]` depth
- **4-dim action space:** 4 thrusters ∈ [-1, 1], 20N max each
- **Fluid physics** applied via `xfrc_applied` (world frame) each substep:
  - Quadratic drag (axial + lateral separately, body frame → world frame via `data.xmat`)
  - Buoyancy offset
  - Water current drag
  - Added mass
- **Reward function** (6 components, all logged to info):
  - progress (×10), alive (×0.1), energy penalty (×0.02), smoothness (×0.05), orientation (×0.5), boundary (×5)
- **DR hook:** `env.randomize_physics(rng)` — all params in `env.physics_params` dict
- `frame_skip=4` → 25Hz control, `max_episode_steps=500` → 20s episodes

### auv_dr_wrapper.py — AUVDomainRandomWrapper
- Wraps HalcyonAUVEnv, intercepts `reset()`, applies DR
- **Three modes in one class:** `mode="none"`, `"uniform"`, `"curriculum"`
- **6 physics parameters randomised:**

| Parameter | CDR start | CDR max (= UDR) | Test held-out |
|-----------|-----------|-----------------|---------------|
| c_drag_lateral | [0.15, 0.25] | [0.10, 0.50] | [0.30, 0.80] |
| c_drag_axial | [0.06, 0.10] | [0.04, 0.20] | [0.12, 0.32] |
| buoyancy_offset | [-0.01, 0.05] | [-0.05, 0.10] | [-0.10, 0.15] |
| current_speed | [0.0, 0.10] | [0.0, 0.30] | [0.20, 0.60] |
| added_mass | [0.10, 0.20] | [0.05, 0.30] | [0.20, 0.50] |
| act_efficiency | [0.95, 1.00] | [0.80, 1.00] | [0.70, 1.00] |

- **CDR mechanism:**
  - Rolling window W=50 episodes
  - Expand ranges when success_rate > 0.70 (70%)
  - Contract ranges when success_rate < 0.40 (40%)
  - Expand/contract step = 5%/3% of full range
  - `curriculum_level` scalar [0,1] tracks expansion progress
- **CDR checkpoint:** `get_cdr_state()` / `load_cdr_state()` → JSON serialisable
- `make_auv_env(xml_path, mode, seed)` — one-liner factory function

### train.py — Full Training Pipeline
- **Auto-detects device:** CUDA (Colab) → MPS (Mac M-series) → CPU
- **Auto-detects save path:** Google Drive if on Colab, `~/rl_research/auv/` locally
- **SAC hyperparameters** (locked in):
  - lr=3e-4, buffer=500k, batch=256, learning_starts=10k, gamma=0.99
  - Network: MLP [256, 256], ReLU, auto entropy tuning
- **VecNormalize:** norm_obs=True, norm_reward=True, clip=10
- **Critical SB3 pattern:** eval env copies `obs_rms` and `ret_rms` from train env, sets `training=False`, `norm_reward=False`
- **Callbacks:** AUVMetricsCallback (TensorBoard), EvalCallback (every 10k steps, 20 episodes), CheckpointCallback (every 50k, saves VecNormalize), CDRCheckpointCallback (saves CDR JSON)
- **CLI:** `python scripts/train.py --mode curriculum --seed 0 --steps 1000000`
- **Saves:** `best_model.zip`, `final_model.zip`, `vec_normalize.pkl`, `cdr_state.json`, `config.json`

### colab_auv_training.ipynb — Proper Jupyter Notebook
- 8 cells: Mount Drive, Install, Clone repo, Sanity check, Train, TensorBoard, Quick eval, CDR inspection
- Upload to Colab via File → Upload notebook (NOT as .py — must be .ipynb)

---

## 4. Current Status (Where We Left Off)

### ✅ Completed
- Full codebase built and validated
- Debug run on Mac: 50k steps, 10.4 min, goal_dist trending DOWN (learning confirmed)
- TensorBoard confirmed working: `train/actor_loss`, `train/critic_loss`, `env/goal_dist`, `cdr/curriculum_level` all logging

### 🔄 Currently Running (DO NOT INTERRUPT)
**3 Colab sessions running in parallel on 3 Google accounts, all T4 GPU:**
- Account 1: `mode=curriculum, seed=0, steps=1_000_000`
- Account 2: `mode=none, seed=0, steps=1_000_000`
- Account 3: `mode=uniform, seed=0, steps=1_000_000`

**Expected completion:** ~4 hours from start (63 it/s on T4)
**All saving to:** `/content/drive/MyDrive/rl_research/auv/<mode>/<mode>_seed0/`

**To keep Colab alive overnight on Mac:**
```bash
caffeinate -i &   # prevents Mac sleep, keeps browser tabs alive
# kill tomorrow with:
killall caffeinate
```

### ⏳ Immediate Next Tasks (when runs finish)
1. **Run Cell 7 (quick eval) on all 3 accounts** — get success rates
2. **Post results in new session** — compare curriculum vs uniform vs none
3. **Start Batch 2:** same 3 accounts, seed=1 for all three modes
4. **Start Batch 3:** seed=2 for all three modes

### 📋 Full Experiment Matrix
| Condition | Seed 0 | Seed 1 | Seed 2 |
|-----------|--------|--------|--------|
| none | 🔄 running | ⏳ pending | ⏳ pending |
| uniform | 🔄 running | ⏳ pending | ⏳ pending |
| curriculum | 🔄 running | ⏳ pending | ⏳ pending |

After 9 main runs: 5 ablation runs (drag only, current only, buoyancy only, mass only, efficiency only) × 3 seeds = 15 more runs.

---

## 5. Next Steps After Runs Finish (in order)

### Step 1 — Collect seed=0 results (immediate)
Run Cell 7 on each Colab account. Record:
- Success rate (primary metric)
- Mean reward ± std
- Mean final goal distance ± std

Expected results (if CDR is working):
- `none`: ~5-15% success (can barely navigate)
- `uniform`: ~30-50% success (learns but fragile)
- `curriculum`: ~50-70% success (robust, generalises)

If CDR is NOT beating uniform at seed=0 → don't panic, check:
1. Is `curriculum_level` > 0 in cdr_state.json? (curriculum must have expanded)
2. Is eval on training distribution or test distribution? (Cell 7 uses training dist)
3. May need 1.5M steps for CDR advantage to emerge

### Step 2 — Build eval.py (held-out test distribution)
This is different from Cell 7. Cell 7 evaluates on the TRAINING distribution.
The paper result requires evaluation on the HELD-OUT TEST distribution (wider ranges).
Need to build `scripts/eval.py` that loads a trained model and runs on test ranges.

### Step 3 — Run seeds 1 and 2
Same setup, 3 parallel accounts, change SEED=1 then SEED=2.

### Step 4 — UUV Simulator setup
Install UUV Simulator (Gazebo-based, ROS). This is the second evaluation environment.
Transfer MuJoCo-trained policy → UUV Simulator for cross-simulator zero-shot eval.
This is the headline result for the paper. Allocate full Week 8.

### Step 5 — Ablation runs
5 conditions × 3 seeds = 15 runs. Each randomises only ONE parameter.
Tells you which parameter drives the sim-to-real gap most.

### Step 6 — PID baseline
Implement a simple PID controller for 3D AUV navigation.
Needed for comparison — every AUV paper includes it.
~1 week to implement, tune, and evaluate.

### Step 7 — Paper writing
Write in this order: Method → Experiments → Results → Figures → Abstract → Introduction → Related Work → Conclusion
See RESEARCH_PLAN.md Section 10 for full writing guide.

---

## 6. Key Technical Decisions (don't revisit these)

### Why body-frame observations?
Policy becomes invariant to world position/heading → necessary for generalisation under current disturbances. This is why the 18-dim obs uses goal_vec_body (not world-frame goal position).

### Why xfrc_applied in world frame?
MuJoCo's `xfrc_applied` is always world frame. Drag computed in body frame then rotated via `data.xmat[body_id].reshape(3,3)`. Getting this wrong would produce drag in the wrong direction when AUV rotates — a silent catastrophic bug.

### Why VecNormalize obs_rms copy pattern?
`VecNormalize.load_from_venv()` does not exist in SB3. Correct pattern: create fresh VecNormalize on eval env, copy `obs_rms` and `ret_rms` from train env, set `training=False`, `norm_reward=False`.

### Why frame_skip=4?
25Hz control rate. Matches realistic AUV thruster update rates. Reduces training time 4× vs frame_skip=1.

### Why CDR window W=50, expand at 70%, contract at 40%?
Based on OpenAI ADR paper (Akkaya 2019). W=50 gives stable success rate estimate without being too slow to react. 70/40 thresholds create a 30-point buffer that prevents oscillation.

---

## 7. Known Bugs Fixed (don't re-introduce these)

1. **`<joint>` in `<default>`** — removed. Was causing implicit free-joint conflicts in MuJoCo. Root joint declared explicitly on halcyon body only.
2. **`VecNormalize.load_from_venv()`** — doesn't exist. Fixed to manual `obs_rms` copy pattern.
3. **`_SCRIPT_DIR / "envs"`** — wrong path. Fixed to `_SCRIPT_DIR.parent / "envs"` since train.py is in `scripts/`, not repo root.
4. **Colab notebook as .py** — Colab expects .ipynb JSON. Built proper notebook file.

---

## 8. Important Files and Where They Are

### On Mac
```
~/rl_robotics/envs/auv.xml
~/rl_robotics/envs/auv_env.py
~/rl_robotics/envs/auv_dr_wrapper.py
~/rl_robotics/scripts/train.py
~/rl_robotics/notebooks/colab_auv_training.ipynb
~/rl_robotics/paper/RESEARCH_PLAN.md
~/rl_research/auv/curriculum/debug_local/   ← debug run results
```

### On Google Drive (Colab results)
```
/content/drive/MyDrive/rl_research/auv/
    curriculum/curriculum_seed0/
        best_model.zip
        final_model.zip
        vec_normalize.pkl
        cdr_state.json
        cdr_state_50000.json  (checkpoint)
        config.json
        tensorboard/
        eval/
    none/none_seed0/           ← currently training
    uniform/uniform_seed0/     ← currently training
```

### GitHub
- Repo: https://github.com/Itzlimon22/Obhyash-complete-project
- All code should be pushed here so Colab can clone it

---

## 9. Papers to Read (priority order)

| Priority | Paper | Where | What to read |
|----------|-------|-------|-------------|
| 🔴 Must | Akkaya 2019 — ADR (OpenAI Rubik's Cube) | arxiv 1910.07113 | Section 3 + Section 5 only |
| 🔴 Must | Tiboni 2024 — DORAEMON | arxiv 2301.12457 | Abstract + Section 3 + Table 1 |
| 🔴 Must | Haarnoja 2018 — SAC | arxiv 1801.01290 | Already using it, read for citations |
| 🟡 Should | MarineGym 2025 — Chu et al. | arxiv, search "MarineGym" | Full paper — closest competitor |
| 🟡 Should | Manhães 2016 — UUV Simulator | GitHub/paper | Setup instructions |
| 🟢 Later | Tobin 2017 — Domain Randomisation | arxiv 1703.06907 | Abstract + intro only |
| 🟢 Later | Fossen 2011 — AUV Hydrodynamics | Textbook | Chapter on drag/buoyancy |

---

## 10. Tone and Style Instructions for AI Assistant

- Researcher is a beginner in RL but has ML basics — explain RL concepts when needed but don't over-explain ML fundamentals
- Always give working, runnable code — no pseudocode unless explicitly asked
- Be opinionated about research decisions — push toward high-impact choices
- When something is wrong, say so directly and fix it immediately
- When asked "what next", give ONE concrete immediate action, not a vague suggestion
- The project deadline is IROS 2026 (March 2, 2026) — always keep this in mind
- Don't rebuild things that are already built — check the file list in Section 3 first
- The codebase is complete. Remaining work is: experiments, eval.py, PID baseline, UUV Simulator setup, paper writing
- Be honest about research quality — this is a good paper but not revolutionary; be precise about what it claims

---

## 11. Commands Reference

```bash
# Activate environment
conda activate rl

# Debug training run (Mac, ~17 min)
cd ~/rl_robotics
python scripts/train.py --mode curriculum --seed 0 --steps 50000 --run-name debug

# Full training run (Mac, ~8 hours — use Colab instead)
python scripts/train.py --mode curriculum --seed 0 --steps 1000000

# TensorBoard (all runs)
tensorboard --logdir ~/rl_research/auv/

# TensorBoard (single run)
tensorboard --logdir ~/rl_research/auv/curriculum/debug_local/tensorboard

# Keep Mac awake for Colab sessions
caffeinate -i &
killall caffeinate   # stop it

# Push to GitHub
cd ~/rl_robotics && git add -A && git commit -m "message" && git push

# Check what's in results directory
ls ~/rl_research/auv/curriculum/
```

---

## 12. What NOT to Do

- ❌ Don't rebuild auv.xml, auv_env.py, auv_dr_wrapper.py, or train.py — they are complete and tested
- ❌ Don't interrupt the 3 currently running Colab sessions
- ❌ Don't change SAC hyperparameters without a specific reason — they are tuned
- ❌ Don't evaluate on training distribution and call it "transfer" — eval.py must use TEST_PARAM_CONFIG
- ❌ Don't overclaim in the paper — this is sim-to-sim transfer, not sim-to-real hardware
- ❌ Don't use `VecNormalize.load_from_venv()` — it doesn't exist in SB3
- ❌ Don't put `<joint>` in `<default>` in MuJoCo XML
- ❌ Don't upload .py files to Colab and expect them to work as notebooks — must be .ipynb

---

*Handoff document generated at end of Session 1.*
*Next session should start by asking: "What were the seed=0 success rates for all three modes?"*
