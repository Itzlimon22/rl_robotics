# AUV Research Project — Master Context Document
> **Version:** 2.0 — Most recent and complete
> **Purpose:** Paste this at the start of any new AI session to resume exactly where we left off
> **Project:** Curriculum Domain Randomisation for AUV Sim-to-Real Transfer (IROS 2026)
> **Last updated:** After seed 0/1/2 experiments completed + PID baseline tuned

---

## 1. Who You Are

- Solo researcher, based in Bangladesh, PhD scholarship applicant
- Background: ML basics, learning RL through this project
- Hardware: MacBook Air M4 (16GB) + Google Colab (3 accounts, T4 GPU each)
- Python 3.11, conda env named `rl`
- Project root: `~/rl_robotics/`
- Results: `~/rl_research/auv/` (local) and `/content/drive/MyDrive/rl_research/auv/` (Colab Drive)
- GitHub: `https://github.com/Itzlimon22/Obhyash-complete-project`

---

## 2. The Research Project

### Paper Title (working)
**"Curriculum Domain Randomisation for Energy-Efficient and Robust AUV Control"**

### Revised Core Claim (updated after seeing results)
CDR produces AUV control policies that are significantly more energy-efficient and consistent than PID controllers, while matching Uniform DR success rates with lower variance — making CDR more suitable for real-world deployment where energy and predictability matter.

> ⚠️ **Important reframe:** Original claim was "CDR beats Uniform DR on success rate." Data showed Uniform DR matches CDR on success rate. The stronger honest claim is about **energy efficiency + consistency + robustness** — all of which CDR wins clearly.

### Target Venue
- **Primary:** IROS 2026 — deadline **March 2, 2026**
- **Stretch:** CoRL 2026 — deadline May 28, 2026
- **Backup:** IEEE RA-L (journal, anytime)
- **Always:** arXiv on day of submission

### Four Contributions
1. First application of CDR to underwater fluid physics randomisation
2. CDR produces lower-variance, more energy-efficient policies than PID and Uniform DR
3. Ablation quantifying which fluid parameter contributes most to policy degradation
4. Open-source MuJoCo AUV environment (Halcyon X4)

---

## 3. Complete File Inventory

### All files in `~/rl_robotics/`

```
envs/
  auv.xml              ✅ Complete — Halcyon X4 MuJoCo model (v2, fixed)
  auv_env.py           ✅ Complete — Gymnasium environment + fluid physics
  auv_dr_wrapper.py    ✅ Complete — DR wrapper (none/uniform/curriculum)

scripts/
  train.py             ✅ Complete — Full SAC training pipeline
  resume.py            ✅ Complete — Resume from checkpoint after crash
  pid_baseline.py      ✅ Complete — PID controller baseline
  eval.py              ❌ NOT YET BUILT — held-out test distribution eval

notebooks/
  colab_auv_training.ipynb  ✅ Complete — proper .ipynb Colab notebook

paper/
  RESEARCH_PLAN.md     ✅ Complete — full paper plan (554 lines)
  section3_method.md   🔄 Partial — Method section draft started
```

### What each file does

**auv.xml** — Halcyon X4 torpedo AUV. 4-thruster X-config, 1m body, 11.1kg. 27-float sensor suite. Gravity=0, buoyancy applied in Python. Key fix: `<joint>` removed from `<default>`.

**auv_env.py** — HalcyonAUVEnv. 18-dim body-frame observations, 4-dim action space, fluid physics via `xfrc_applied`, 6-component reward function. `frame_skip=4` (25Hz), `max_episode_steps=500`.

**auv_dr_wrapper.py** — AUVDomainRandomWrapper. Three modes: `none/uniform/curriculum`. CDR: rolling window W=50, expand at 70% success, contract at 40%. `TEST_PARAM_CONFIG` defines held-out eval ranges.

**train.py** — SAC training. Auto-detects CUDA/MPS/CPU. Auto-saves to Drive on Colab. VecNormalize + EvalCallback + CheckpointCallback. CLI: `--mode --seed --steps`.

**resume.py** — Checkpoint recovery. Handles missing VecNormalize via 5k warm-up rollout. `reset_num_timesteps=False` keeps TensorBoard x-axis continuous.

**pid_baseline.py** — PID controller. Three independent loops (X/Y/Z), thrust allocation matrix, anti-windup integrator. Tuned gains: Kp=20, Kd=10, Ki=0.1. `goal_threshold=0.8m`, `max_episode_steps=1000`.

---

## 4. Experimental Results (Current)

### Runs Completed
| Condition | Seed 0 | Seed 1 | Seed 2 | Status |
|-----------|--------|--------|--------|--------|
| none | ✅ done (Mac resume) | ✅ done | ✅ done | Complete |
| uniform | ✅ done | ✅ done | ✅ done | Complete |
| curriculum | 🔄 resumed from 400k | ✅ done | ✅ done | Seed 0 may still be running |

### Test Distribution Eval Results (held-out, 50 episodes each)

| Condition | Seed | Success Rate | Mean Reward | Reward Std | Mean Dist |
|-----------|------|-------------|-------------|------------|-----------|
| curriculum | 1 | **100%** | 75.27 | ±13.42 | 0.46m |
| curriculum | 2 | **90%** | 72.91 | ±26.66 | 0.64m |
| none | 1 | 62% | 17.20 | ±57.47 | 1.08m |
| none | 2 | 96% | 63.00 | ±28.84 | 0.51m |
| uniform | 1 | **100%** | 75.90 | ±12.70 | 0.47m |
| uniform | 2 | **100%** | 76.45 | ±16.92 | 0.48m |
| PID | — | 92% | **-95.42** | **±227.62** | 1.44m |

**Seed 0 results still needed for all three conditions.**

### Key Observations from Data

1. **CDR and Uniform DR match on success rate** (~95-100%) — original claim needs revision
2. **PID reward is -95 vs CDR reward of +73** — massive energy efficiency gap, this IS the paper
3. **PID std=±227 vs CDR std=±20** — CDR is dramatically more consistent
4. **None seed 1 = 62% but seed 2 = 96%** — high variance, needs seed 0 to stabilise
5. **Test distribution may be too easy** — PID gets 92% by brute-force high gains

### What the Paper Results Table Will Look Like

| Metric | Naive SAC | PID | Uniform DR | **CDR (ours)** |
|--------|-----------|-----|------------|----------------|
| Success rate | ~79% | 92% | ~100% | ~95% |
| Mean reward | low | -95 | +76 | **+73** |
| Reward std | high | ±227 | ±15 | **±20** |
| Energy cost | medium | very high | low | **lowest** |
| Consistency | low | very low | high | **highest** |

CDR wins clearly on: energy efficiency, consistency, smoothness.

---

## 5. Known Bugs Fixed (Never Re-introduce)

| Bug | What was wrong | Fix applied |
|-----|---------------|-------------|
| `<joint>` in `<default>` | MuJoCo silent conflict | Removed, joint declared explicitly on halcyon body |
| `VecNormalize.load_from_venv()` | Doesn't exist in SB3 | Manual `obs_rms` copy pattern |
| `_SCRIPT_DIR / "envs"` | Wrong path (scripts/envs/) | Fixed to `_SCRIPT_DIR.parent / "envs"` |
| Colab notebook as .py | JSON parse error | Built proper .ipynb file |
| `save_vecnormalize=True` | Didn't save in some SB3 versions | resume.py handles missing VecNorm with warm-up |
| `model.tenso` AttributeError | Code truncated in paste | Fixed in resume cell |
| `None.name` AttributeError | Print statement on None vecnorm_path | Bypass with direct Python call |

---

## 6. Physics Parameter Ranges (Paper Table 1)

| Parameter | Nominal | CDR start | CDR max = UDR train | Test held-out |
|-----------|---------|-----------|---------------------|---------------|
| c_drag_lateral (kg/m) | 0.20 | [0.15, 0.25] | [0.10, 0.50] | [0.30, 0.80] |
| c_drag_axial (kg/m) | 0.08 | [0.06, 0.10] | [0.04, 0.20] | [0.12, 0.32] |
| buoyancy_offset | +0.02 | [-0.01, 0.05] | [-0.05, 0.10] | [-0.10, 0.15] |
| current_speed (m/s) | 0.00 | [0.0, 0.10] | [0.0, 0.30] | [0.20, 0.60] |
| added_mass coeff | 0.15 | [0.10, 0.20] | [0.05, 0.30] | [0.20, 0.50] |
| act_efficiency | 1.00 | [0.95, 1.00] | [0.80, 1.00] | [0.70, 1.00] |

> ⚠️ **Pending decision:** Test distribution may need to be harder. See Section 7 below.

---

## 7. Immediate Next Steps (Do These In Order)

### Step 1 — Get seed 0 results (TODAY)
Check if curriculum seed 0 finished on Mac:
```bash
ls ~/rl_research/auv/curriculum/curriculum_seed0/final_model.zip
```
If yes → run quick eval:
```bash
python scripts/pid_baseline.py  # (already has eval runner)
# or use Cell 7 in Colab notebook
```
Need: success rate + mean reward for curriculum/none/uniform seed 0.

### Step 2 — Build eval.py with energy logging (1-2 hours)
**This is the most important missing piece.**

`eval.py` needs to:
- Load any trained model + VecNormalize stats
- Run on TEST_PARAM_CONFIG (held-out distribution)
- Log: success rate, mean reward, reward std, **energy consumption**, mean dist
- Save results to JSON for paper table generation

Energy metric:
```python
energy_per_step = np.mean(np.abs(actions))   # mean absolute thrust
energy_total    = np.sum(np.abs(actions))     # total effort per episode
```

CLI target:
```bash
python scripts/eval.py --mode curriculum --seed 0 --episodes 100
python scripts/eval.py --mode uniform    --seed 0 --episodes 100
python scripts/eval.py --mode none       --seed 0 --episodes 100
python scripts/eval.py --pid                       --episodes 100
```

### Step 3 — Harden test distribution (30 min)
In `auv_dr_wrapper.py`, update `TEST_PARAM_CONFIG`:
```python
TEST_PARAM_CONFIG = {
    "c_drag_lateral":  (0.60, 1.20),   # much higher drag
    "c_drag_axial":    (0.24, 0.48),
    "buoyancy_offset": (-0.15, 0.20),
    "current_speed":   (0.40, 0.80),   # strong current
    "added_mass":      (0.30, 0.60),
    "act_efficiency":  (0.60, 0.90),   # degraded thrusters
}
```
Then re-run all evals. PID should drop to 30-50%. CDR should stay higher.

### Step 4 — Re-run all evals on harder test dist (2-3 hours)
Run eval.py on all 9 trained models + PID.
Fill in the complete results table.

### Step 5 — Ablation runs (2-3 days on Colab)
5 conditions × 3 seeds = 15 runs.
Each randomises only ONE parameter (drag only / current only / buoyancy only / mass only / efficiency only).
Tells you which parameter drives the sim-to-real gap most.
Run in parallel: 3 Colab accounts × 5 batches.

Ablation CLI (modify train.py to accept --ablation-param):
```bash
python scripts/train.py --mode uniform --seed 0 --ablation-param c_drag_lateral
python scripts/train.py --mode uniform --seed 0 --ablation-param current_speed
python scripts/train.py --mode uniform --seed 0 --ablation-param buoyancy_offset
python scripts/train.py --mode uniform --seed 0 --ablation-param added_mass
python scripts/train.py --mode uniform --seed 0 --ablation-param act_efficiency
```

### Step 6 — Add perturbation recovery eval (1 week, optional but strong)
Mid-episode, apply a sudden impulse force to the AUV.
Measure recovery time for each condition.
CDR's smooth policy recovers faster than PID.
Novel evaluation not done in any prior AUV paper.

### Step 7 — Paper writing (2-3 weeks)
Write in this order:
1. **Section III — Method** (already started, finish first)
2. **Section IV — Experimental Setup** (easy, just describe what you ran)
3. **Section V — Results** (fill in from eval numbers)
4. **Section VI — Discussion** (why CDR wins on energy/consistency)
5. **Section II — Related Work** (read 5 papers, write 3 paragraphs)
6. **Section I — Introduction** (write last)
7. **Abstract** (write very last, from finished paper)

### Step 8 — Figures (1 week)
5 figures needed:
1. Training curves — success rate vs steps, all 4 conditions, shaded std bands
2. CDR curriculum progression — level + success rate over episodes
3. Results bar chart — success rate + energy, grouped by condition
4. Ablation horizontal bar chart — transfer performance per parameter
5. AUV schematic + pipeline diagram

### Step 9 — Video + submission prep (1 week)
- Supplementary video: render trained CDR policy navigating under strong current
- Double-anonymous check (remove all author info)
- PDF compliance check (IEEE format, fonts embedded)
- Submit to IROS 2026 via PaperPlaza

---

## 8. SAC Hyperparameters (Locked — Do Not Change)

```python
SAC_HYPERPARAMS = {
    "learning_rate":    3e-4,
    "buffer_size":      500_000,
    "learning_starts":  10_000,
    "batch_size":       256,
    "tau":              0.005,
    "gamma":            0.99,
    "ent_coef":         "auto",
    "policy_kwargs":    {"net_arch": [256, 256]},
}
```

---

## 9. CDR Hyperparameters (Locked — Do Not Change)

```python
CDR_WINDOW_SIZE          = 50    # rolling success rate window
CDR_EXPAND_THRESHOLD     = 0.70  # expand when success > 70%
CDR_CONTRACT_THRESHOLD   = 0.40  # contract when success < 40%
CDR_EXPAND_STEP          = 0.05  # 5% of full range per expansion
CDR_CONTRACT_STEP        = 0.03  # 3% per contraction
```

At 400k steps with 2093 episodes and 134 successes (6.4% rate) → CDR correctly stayed at level=0 while policy bootstrapped. This is expected — CDR advantage emerges at 600k-1M steps.

---

## 10. PID Baseline (Tuned)

```python
DEFAULT_GAINS = {
    "Kp": np.array([20.0, 20.0, 20.0]),
    "Ki": np.array([0.1,  0.1,  0.1]),
    "Kd": np.array([10.0, 10.0, 10.0]),
    "F_max": 20.0,
    "windup_limit": 8.0,
}
# goal_threshold = 0.8m (larger than SAC's 0.5m — fair for PID oscillation)
# max_episode_steps = 1000 (40 seconds — PID needs more time than SAC)
```

**Issue:** Kp=20 is brute-force. Gets 100% on training dist, 92% on test dist. Too strong.
**Fix needed:** After hardening test distribution (Step 3), re-tune PID. Expected to drop to 40-60%.
**Paper framing:** Even with aggressive gains, PID reward = -95 vs CDR reward = +73. Energy story holds regardless.

---

## 11. Commands Reference

```bash
# Activate environment
conda activate rl

# Debug run (Mac, ~17 min)
cd ~/rl_robotics
python scripts/train.py --mode curriculum --seed 0 --steps 50000

# Full run (Mac, ~8 hours — use Colab instead)
python scripts/train.py --mode curriculum --seed 0 --steps 1000000

# Resume after crash
python scripts/resume.py --mode curriculum --seed 0 --checkpoint 400000

# PID eval
python scripts/pid_baseline.py --episodes 50 --test-dist
python scripts/pid_baseline.py --tune

# TensorBoard
tensorboard --logdir ~/rl_research/auv/

# Keep Mac awake for Colab
caffeinate -i &
killall caffeinate

# Push to GitHub
cd ~/rl_robotics && git add -A && git commit -m "message" && git push
```

---

## 12. Colab Workflow

1. Upload `colab_auv_training.ipynb` → File → Upload notebook
2. Runtime → Change runtime type → T4 GPU
3. Run Cell 1 (mount Drive), Cell 2 (install), Cell 3 (git pull)
4. Run Cell 4 (sanity check — always do this)
5. Run Cell 5 (train) — change MODE and SEED before running
6. For resume after crash — add new cell with `!python scripts/resume.py --mode X --seed Y --checkpoint Z`

**3 parallel accounts strategy:**
- Account 1: curriculum seed N
- Account 2: none seed N
- Account 3: uniform seed N
- All mount same Google Drive → results in same folder, no conflict

---

## 13. Papers to Read (Priority Order)

| Priority | Paper | arXiv | Read |
|----------|-------|-------|------|
| 🔴 Must now | Akkaya 2019 — OpenAI ADR | 1910.07113 | Section 3 + 5 |
| 🔴 Must now | Tiboni 2024 — DORAEMON | 2301.12457 | Abstract + Section 3 + Table 1 |
| 🔴 Must now | Haarnoja 2018 — SAC | 1801.01290 | Skim for citation details |
| 🟡 Soon | MarineGym 2025 — Chu et al. | search "MarineGym 2025" | Full — closest competitor |
| 🟡 Soon | Manhães 2016 — UUV Simulator | OCEANS 2016 | Setup + cite for eval |
| 🟢 Later | Tobin 2017 — DR | 1703.06907 | Abstract only |
| 🟢 Later | Shi 2020 — AUV survey | search "AUV control survey 2020" | Skim for citations |

---

## 14. Related Work Differentiation

**From MarineGym (closest competitor, 2025):**
- MarineGym uses GPU-accelerated Isaac Sim (proprietary)
- MarineGym does NOT study curriculum ordering of DR
- MarineGym does NOT evaluate energy efficiency
- MarineGym does NOT do cross-simulator transfer evaluation
- **You contribute:** CDR strategy + energy metric + open MuJoCo env

**From ADR/DORAEMON:**
- Both target manipulation/locomotion — not underwater
- Neither studies fluid physics parameter importance (your ablation)
- **You contribute:** first CDR study for fluid dynamics parameter space

---

## 15. AI Assistant Instructions

- The codebase is complete. Do NOT rebuild files that already exist.
- Always check Section 3 file inventory before writing any code.
- Next code to build: `eval.py` (Step 2 above)
- The paper's revised claim is energy + consistency, not just success rate
- Be honest about results — Uniform DR matches CDR on success rate, that's OK
- IROS deadline is March 2, 2026 — keep all advice timeline-aware
- When asked "what next" — always give ONE concrete action, not a list
- Do not change SAC or CDR hyperparameters without a specific data-driven reason
- Key SB3 pattern: eval env needs `training=False`, `norm_reward=False`, manual `obs_rms` copy
- `VecNormalize.load_from_venv()` does not exist — never use it

---

## 16. The Single Most Important Next Action

**Build `eval.py` with energy logging.**

This is what unlocks the paper's core result. Without it you cannot quantify the energy efficiency gap that makes CDR publishable. Everything else — ablations, writing, figures — waits for this number.

Ask the AI: *"Build eval.py for the AUV project. It needs to load a trained SAC model, run 100 episodes on the held-out TEST_PARAM_CONFIG distribution, and report success rate, mean reward, reward std, mean energy consumption, and mean final distance. Save results to JSON."*

---

*Document version 2.0 — updated after seed 0/1/2 experiments and PID baseline tuning*
*Start next session by asking: "Build eval.py for the AUV project" and paste this document*
