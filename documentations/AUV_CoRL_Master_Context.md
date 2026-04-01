# AUV Project Master Context & Post-1M Execution Plan
**Date:** April 2026  
**Target Venue:** CoRL 2026 (May 28 Deadline)  
**Project:** Curriculum Domain Randomisation (CDR) for Energy-Efficient AUV Sim-to-Real Transfer  

## Instructions for AI Assistant
Act as an expert robotics, reinforcement learning, and systems engineering research assistant. I am transferring context from a previous development session. The codebase is currently 100% complete and bug-free. 
**Do NOT write any new environment or wrapper code.** Your role in this session is strictly:
1. Data analysis and visualization of my 1M-step evaluation results.
2. Drafting the CoRL 2026 paper (Methodology, Results, Abstract).

Read the context below, acknowledge it, and ask me to provide the 1M-step evaluation data.

---

## 1. Finalized Codebase Architecture
We successfully built a custom MuJoCo/Gymnasium simulation pipeline using Stable Baselines 3 (SAC). The training stack consists of 5 integrated phases:
* **Phase 0 (Energy Logging):** Bypassed SB3's default evaluation to log **mean absolute thrust** (energy consumption), the core claim of our paper.
* **Phase 1 (Sensor Noise):** Injected Gaussian noise ($\mathcal{N}(0, \sigma^2)$) into positional and velocity observations. Noise limits are scaled by the CDR wrapper.
* **Phase 2 (Dynamic Goal):** Goal marker drifts at 0.3 m/s and bounces off workspace boundaries.
* **Phase 3 (Trajectory Tracking):** AUV follows a 3D lemniscate path using a 3D predictive lookahead vector rotated into the local body frame via quaternion conjugation.
* **Phase 4 (Obstacle Avoidance):** 23-dimension observation space. Base 19-dim + 4 rangefinder sensors (forward, port, starboard, up). Environment spawns 5 spherical obstacles randomly each episode.
* **Phase 5 (Sensor Glitch Evaluation):** A custom mid-episode perturbation script (`eval_perturbation.py`) that blinds the AUV's rangefinders with severe noise for 10 steps to test fault-tolerant recovery.

## 2. The 50k-Step Baseline Data (Crucial for Figure 3)
Before launching the 1,000,000-step training runs, we tested a 50,000-step "baby" CDR model on the Phase 5 Sensor Glitch Perturbation.
**The 50k Results:**
* **Curriculum Expansion:** Reached level `0.722` (72% of max physics difficulty).
* **Network Health:** `actor_loss` stabilized (-0.87), `critic_loss` collapsed to near-zero (0.00058), entropy decayed appropriately.
* **Sensor Glitch Test:** 0.0% Success Rate | **131.9 Mean Recovery Steps**.
* *Significance:* This 131.9 step survival time is our pre-converged baseline. It proves the AUV didn't instantly die when blinded; it fought to survive using spatial memory/momentum but eventually drifted too far. 

## 3. The 1M-Step Evaluation Protocol
I have three models currently finishing a 1,000,000-step training run on Google Colab:
1. `mode="none"` (Naive SAC - fixed nominal physics)
2. `mode="uniform"` (Standard Domain Randomization)
3. `mode="curriculum"` (Our novel CDR approach)

**The Immediate Task:**
Run `eval_perturbation.py` on all three 1M-step models. 
* *Hypothesis:* The Naive model will crash shortly after the glitch. The Uniform model will perform adequately but with high variance. The CDR model will survive the 10-step glitch, recover efficiently, and successfully reach the goal, proving CDR yields highly robust spatial memory and fault tolerance.

## 4. Paper Writing Strategy (8-Week Sprint)
With the May 28 CoRL deadline approaching, development is frozen. The current focus is manuscript generation.
* **Figure 3 Anchor:** A bar/violin plot comparing "Mean Survival Steps Under Sensor Failure" (50k CDR vs. 1M Naive vs. 1M Uniform vs. 1M CDR).
* **Core Narrative:** CDR produces AUV policies that are not just successful in varied currents, but physically tougher and more energy-efficient than baseline RL and traditional PID controllers.

---
**End of Context Transfer.** ```

***

### How to use this:
1. Save it to your machine.
2. Let your Colab instances finish their 1,000,000-step runs.
3. Download the `final_model.zip` and `vec_normalize.pkl` files from Colab to your MacBook.
4. Run your `eval_perturbation.py` script on all three models (None, Uniform, Curriculum).
5. Open a new AI session, paste that Markdown document, and then paste your new evaluation numbers. We will instantly start writing your CoRL paper!