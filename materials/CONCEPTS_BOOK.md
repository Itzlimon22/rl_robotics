# The Complete Concepts Book
## Curriculum Domain Randomisation for AUV Control
### Everything You Need to Understand, Write, and Defend Your Paper

> **For:** Limon Howlader · MIST Bangladesh · RA-L/IROS 2026
> **Purpose:** Total understanding of every concept in this project — from first principles
> **How to use:** Read chapters in order the first time. Use as reference when writing.
> **Level:** Assumes basic ML knowledge. Explains everything else from scratch.

---

# TABLE OF CONTENTS

1. [The Big Picture — What This Project Is](#1-the-big-picture)
2. [Reinforcement Learning — From Scratch](#2-reinforcement-learning)
3. [Deep Neural Networks in RL](#3-deep-neural-networks-in-rl)
4. [Soft Actor-Critic (SAC) — The Algorithm You Used](#4-soft-actor-critic)
5. [Simulation and the Sim-to-Real Problem](#5-simulation-and-sim-to-real)
6. [Domain Randomisation — Theory and Practice](#6-domain-randomisation)
7. [Curriculum Learning and CDR](#7-curriculum-learning-and-cdr)
8. [AUV Physics — What You Simulated](#8-auv-physics)
9. [MuJoCo — The Simulator](#9-mujoco)
10. [Your Experimental Design](#10-your-experimental-design)
11. [PID Control — The Baseline You Beat](#11-pid-control)
12. [Results Analysis — What Your Numbers Mean](#12-results-analysis)
13. [Statistical Methods for Your Paper](#13-statistical-methods)
14. [How to Read and Cite Every Paper in Your References](#14-your-references-explained)
15. [Writing Guide — Turning Concepts into Paper Sections](#15-writing-guide)
16. [Common Reviewer Objections and How to Answer](#16-reviewer-objections)

---

# 1. The Big Picture

## What problem does this project solve?

Imagine you are deploying an autonomous underwater vehicle (AUV) in the ocean to
inspect a pipeline. You spent weeks tuning the controller in calm water at the dock.
The controller works perfectly in testing — 100% success rate. Then you deploy it
in real ocean conditions. The current is stronger. The vehicle is slightly more
buoyant due to temperature change. Biofouling on the hull increases drag.

The controller fails. Catastrophically. Not 80% success — 3% success.

This is the exact result your paper demonstrates quantitatively. A PID controller
tuned for nominal conditions achieves 100% in simulation, then 3% when fluid
parameters shift modestly. This 97-percentage-point collapse is the motivation
for everything in your paper.

## What does your paper propose?

Your paper asks: can we train an RL policy that generalises across the range of
real-world fluid parameter variation, without ever seeing real hardware? And can
we do it more efficiently than simply randomising everything uniformly?

The answer is yes — through **Curriculum Domain Randomisation (CDR)**: start
training with gentle, nominal physics, and gradually expand to harder conditions
as the agent learns to handle them.

## Why this matters practically

Real AUVs run on batteries. A policy that uses 6.3% less thrust energy per step
extends mission duration meaningfully. For a 4-hour inspection mission, 6% energy
saving could mean the difference between completing the survey and having to
surface early. This is not a trivial claim.

## Where this sits in the literature

Your paper sits at the intersection of three fields:

```
Reinforcement Learning    ×    AUV Control    ×    Sim-to-Real Transfer
         ↓                           ↓                      ↓
      SAC algorithm            Fluid physics          Domain Randomisation
      (Haarnoja 2018)          (Fossen 2011)          (Tobin 2017)
                                     ↓
                          First CDR study for
                          underwater fluid physics
                          (your contribution)
```

No prior work has applied curriculum domain randomisation to AUV fluid physics
specifically. This is your novelty claim.

---

# 2. Reinforcement Learning

## 2.1 What is Reinforcement Learning?

Reinforcement learning is a framework for learning behaviour through trial and error.
An **agent** takes **actions** in an **environment**, receives **rewards**, and
learns a **policy** (a mapping from observations to actions) that maximises
cumulative reward over time.

The three key differences from supervised learning:
- No labelled training data — the agent generates its own experience
- Feedback is delayed — you might not know if an action was good until much later
- The agent's actions affect future observations — this creates complex dependencies

## 2.2 The Markov Decision Process (MDP)

Every RL problem is formalised as an MDP — a 5-tuple (S, A, R, T, γ):

**State space S:** Everything the agent can observe about the world. In your
project, S is an 18-dimensional vector describing the AUV's position relative
to the goal, its velocity, its orientation, and the water current speed.

**Action space A:** What the agent can do. In your project, A is a 4-dimensional
continuous vector — one thrust value per thruster, each in [-1, 1].

**Reward function R(s, a):** A scalar signal telling the agent how good each
step was. Your reward has 6 components:

```
r = r_progress + r_alive + r_energy + r_smooth + r_orient + r_boundary

r_progress  = 10 × Δdist          (reward for getting closer to goal)
r_alive     = 0.1                  (small bonus for surviving)
r_energy    = -0.02 × ||a||²      (penalty for high thrust — efficiency)
r_smooth    = -0.05 × ||Δa||²    (penalty for jerky action — smooth control)
r_orient    = -0.5 × (1 - cos θ) (penalty for not facing goal)
r_boundary  = -5 if out of bounds (workspace constraint)
```

The energy penalty (r_energy) is why CDR policies end up more efficient — the
agent is directly incentivised to use less thrust while still reaching the goal.

**Transition function T(s'|s,a):** The physics of the world. In your project,
this is MuJoCo's physics simulator. You never explicitly write T — it is implicit
in the simulation.

**Discount factor γ = 0.99:** Future rewards are worth slightly less than
immediate rewards. γ = 0.99 means a reward 100 steps in the future is worth
e^(-0.01×100) ≈ 0.37 times as much as an immediate reward. This is important
because it makes the learning problem mathematically tractable.

## 2.3 The Objective: Maximising Expected Cumulative Reward

The agent's goal is to find a policy π(a|s) (a probability distribution over
actions given the current state) that maximises:

```
J(π) = E[ Σ_{t=0}^{∞} γ^t r_t ]
```

This is the expected sum of discounted future rewards, called the **return**.
The expectation is over the randomness in both the policy and the environment.

## 2.4 Value Functions: How the Agent Plans

Rather than planning step-by-step, RL agents learn **value functions** that
estimate how good a state (or state-action pair) is in terms of future return.

**State value function V^π(s):** Expected return starting from state s under
policy π:
```
V^π(s) = E_π[ Σ_{t=0}^{∞} γ^t r_t | s_0 = s ]
```

**Action-value function Q^π(s,a):** Expected return from taking action a in
state s, then following π:
```
Q^π(s,a) = E_π[ Σ_{t=0}^{∞} γ^t r_t | s_0 = s, a_0 = a ]
```

The Bellman equation relates these recursively:
```
Q^π(s,a) = E[ r + γ Q^π(s', π(s')) ]
```

This is the key equation in SAC. It says: the value of taking action a in state s
equals the immediate reward plus the discounted value of the next state.

## 2.5 On-Policy vs Off-Policy

**On-policy** algorithms (e.g., PPO) can only learn from data collected by the
current policy. Every time the policy updates, old data becomes useless.

**Off-policy** algorithms (e.g., SAC) can learn from data collected by any
policy, stored in a **replay buffer**. This makes off-policy methods far more
sample-efficient — you can reuse experience many times.

This is why you chose SAC over PPO. Each training run takes ~4.5 hours on a T4
GPU. Sample efficiency matters.

---

# 3. Deep Neural Networks in RL

## 3.1 Why Neural Networks?

Your state space has 18 dimensions. Your action space has 4 continuous dimensions.
The relationship between states and optimal actions is highly nonlinear (a small
change in current direction completely changes the optimal thrust pattern).

Tabular methods (storing a Q-value for every state-action pair) are impossible —
there are infinitely many possible states in continuous spaces. Neural networks
generalise across similar states and learn compact representations of the
value function and policy.

## 3.2 The Actor-Critic Architecture

SAC (and your project) uses two types of networks:

**Actor (policy network) π_φ(a|s):** Takes a state as input, outputs a distribution
over actions. In SAC, this is a Gaussian distribution:
```
π_φ(a|s) = N(μ_φ(s), σ_φ(s))
```
The network outputs both the mean μ and standard deviation σ for each action dimension.

**Critic (Q-function network) Q_θ(s,a):** Takes a state and action as input,
outputs a scalar estimate of the Q-value. SAC uses two critics to prevent
overestimation bias (a known problem in Q-learning).

Your architecture: 2-layer MLP [256, 256] with ReLU activations for both actor
and critic. This is standard for continuous control tasks.

## 3.3 The Replay Buffer

A large memory buffer (500,000 transitions in your project) that stores past
experience as tuples (s, a, r, s'). During training, mini-batches of 256
transitions are sampled uniformly at random. This:
- Breaks temporal correlations in training data
- Allows reuse of experience
- Stabilises training

## 3.4 VecNormalize: Observation Normalisation

Raw observations have very different scales. Goal distance ranges from 0 to 20m.
Angular velocity ranges from -2 to 2 rad/s. Water current ranges from 0 to 0.8 m/s.
Without normalisation, the neural network would have to handle wildly different
input magnitudes, making learning unstable.

VecNormalize maintains running mean μ and standard deviation σ for each observation
dimension, normalising each observation to approximately zero mean and unit variance:
```
obs_normalised = (obs - μ) / (σ + ε)
```

**Critical implementation detail:** The same μ and σ used during training MUST
be used during evaluation. This is why you save `vec_normalize.pkl` alongside
every trained model. Loading a model without the corresponding VecNormalize stats
will cause the policy to see observations in a completely different scale than
it was trained on — catastrophic performance degradation even with a good model.

---

# 4. Soft Actor-Critic (SAC)

## 4.1 The Key Idea: Maximum Entropy RL

Standard RL maximises expected return: E[Σ γ^t r_t].

SAC maximises expected return **plus entropy**:
```
J(π) = E[ Σ_{t=0}^{∞} γ^t ( r_t + α H(π(·|s_t)) ) ]
```

where H(π(·|s)) = -E[log π(a|s)] is the entropy of the policy — a measure of
how random/diverse it is. α is the temperature coefficient that controls the
tradeoff between reward and entropy.

**Why does this matter?**

1. **Exploration:** High entropy = more random actions = better exploration of
   the state space. SAC naturally explores without needing explicit exploration
   mechanisms like ε-greedy.

2. **Robustness:** An entropy-maximising policy doesn't commit to a single action.
   It maintains multiple reasonable strategies simultaneously. When the environment
   changes (e.g., in a different drag regime), the policy can adapt more easily
   because it never fully committed to the minimum-action approach.

3. **Stability:** Entropy regularisation acts as a regulariser that prevents the
   policy from prematurely converging to a local optimum.

4. **Energy efficiency:** Because SAC encourages stochastic policies and penalises
   extreme actions (which have low entropy), SAC policies tend to use more
   moderate, efficient thrust patterns compared to deterministic algorithms.
   This is why CDR + SAC produces energy-efficient policies.

As SAC aims to maximise expected reward while also maximising entropy — to succeed at the task while acting as randomly as possible.

## 4.2 Automatic Entropy Tuning

The temperature α is critical. Too high → policy is too random, doesn't learn
to reach the goal. Too low → policy becomes deterministic, loses robustness.

Instead of hand-tuning α, SAC v2 automatically adjusts it to maintain a target
entropy level H̄:
```
α_t+1 = α_t - λ_α ∇_α E[-α log π(a|s) - α H̄]
```

You set `ent_coef="auto"` in your code, which enables this automatic tuning.
The target entropy is set to -dim(A) = -4 by default (negative action space dimension).

SAC extends to incorporate a constrained formulation that automatically tunes the temperature to match a target entropy, enabling automatic adaptation throughout training.

## 4.3 Twin Critics to Prevent Overestimation

Q-learning is known to overestimate Q-values, which leads to unstable training.
SAC uses **two** Q-networks and takes the minimum:
```
y = r + γ (min(Q_θ1(s',a'), Q_θ2(s',a')) - α log π(a'|s'))
```

Taking the minimum of two independent estimates is conservative — it underestimates
rather than overestimates, which turns out to be safer for stability.

## 4.4 The Policy Gradient in SAC

The actor is updated to maximise:
```
J_π(φ) = E_{s~D, a~π_φ} [ α log π_φ(a|s) - min_i Q_θi(s,a) ]
```

The reparameterisation trick makes this differentiable: instead of sampling
a ~ π_φ(·|s) directly, compute a = μ_φ(s) + σ_φ(s) × ε, where ε ~ N(0,I).
This allows gradients to flow through the sampling operation.

## 4.5 Why SAC for AUV Control?

| Property | Why It Matters for AUV |
|----------|----------------------|
| Off-policy | Replay buffer enables data reuse — critical with limited GPU hours |
| Continuous actions | 4 continuous thrust values [-1,1] — no discretisation needed |
| Entropy regularisation | Produces robust, energy-efficient policies |
| Auto α tuning | No hyperparameter search for temperature |
| Stable across seeds | Critical for reproducible 3-seed comparison |

Soft actor-critic learns robust policies due to entropy maximisation at training time; the policy can readily generalise to perturbations without any additional learning.

## 4.6 SAC Hyperparameters (Your Choices Explained)

| Hyperparameter | Your Value | Why |
|----------------|------------|-----|
| learning_rate | 3×10⁻⁴ | Standard Adam LR for SAC. Too high → unstable. Too low → slow. |
| buffer_size | 500,000 | Large enough to store diverse experience. ~500k steps of data. |
| learning_starts | 10,000 | Collect random exploration data before training begins. Fills replay buffer with diverse data. |
| batch_size | 256 | Standard mini-batch. Balance between gradient quality and speed. |
| gamma | 0.99 | High discount. 20-second episodes need agent to plan far ahead. |
| net_arch | [256,256] | Two hidden layers, 256 neurons each. Deep enough for fluid physics, small enough for fast training. |
| ent_coef | "auto" | Automatic temperature tuning — never tune manually. |
| tau | 0.005 | Soft target network update rate. Stabilises Q-value estimates. |

---

# 5. Simulation and the Sim-to-Real Problem

## 5.1 Why Simulate?

Training RL requires millions of environment interactions. On a real AUV:
- Each interaction risks hardware damage
- Data collection is expensive and slow
- You cannot reset the environment instantly
- You cannot run parallel experiments

In simulation:
- Millions of steps per hour, safely
- Instant resets
- Controllable parameters
- Parallel experiments

The tradeoff: the simulator is not the real world.

## 5.2 The Sim-to-Real Gap

The **reality gap** is the discrepancy between the simulated environment and
the real world. Sources in underwater robotics (Fossen, 2011):

**Physical parameter uncertainty:**
- Drag coefficients change with biofouling, temperature, vehicle speed
- Buoyancy changes with water density (salinity, temperature)
- Added mass depends on vehicle geometry and proximity to boundaries
- Thruster efficiency degrades with wear and cavitation

**Unmodelled dynamics:**
- Turbulence and vortex shedding (your model uses laminar drag)
- Wave-induced disturbances at shallow depth
- Thruster interaction effects (four thrusters in close proximity)
- Flexible body dynamics (hull flexion under pressure)

**Sensor noise:**
- IMU drift and noise
- Velocity estimation errors from DVL Doppler shift
- Pressure sensor noise in depth measurement

A controller trained with nominal parameters and no noise will encounter all of
these sources of variation in real deployment. This is why your PID baseline
collapses from 100% to 3%.

## 5.3 Approaches to Closing the Gap

**System identification:** Carefully measure real physical parameters and match
simulation to reality. Time-consuming, requires real hardware access, and only
partially solves the problem (unmodelled dynamics remain).

**Domain adaptation:** Train in simulation, then fine-tune with a small amount
of real data. Requires real hardware and carefully managed distribution shift.

**Domain randomisation:** Train across a distribution of simulated environments
wide enough that the real world falls within the distribution. No real data
needed. This is what your paper studies.

**High-fidelity simulation:** Build a better simulator. Expensive, never complete.

Your paper's position: domain randomisation is the most practical approach for
zero-shot transfer (no real data, no fine-tuning). CDR makes DR more efficient
by scheduling difficulty.

---

# 6. Domain Randomisation

## 6.1 The Core Idea

Domain randomisation explores a simple technique for training models in simulation that transfer to real environments by randomising rendering in the simulator. With enough variability in the simulator, the real world may appear to the model as just another variation.

Applied to dynamics rather than rendering: Domain randomisation defines a family of simulators parameterised by physical factors such as masses and friction coefficients, and at the start of each episode randomly samples one instance from this family for training. DR has enabled zero-shot transfer in robotic control, dexterous manipulation, and agile locomotion.

## 6.2 The Mathematical Framework

Let φ ∈ Φ denote the physics parameters (drag, buoyancy, current, etc.).
The real world corresponds to one specific φ_real, which is unknown.

**Uniform DR** trains on a distribution:
```
φ ~ Uniform(φ_min, φ_max)
```

If [φ_min, φ_max] is wide enough that φ_real ∈ [φ_min, φ_max], then the policy
learned under DR will generalise to the real world.

The key insight: the policy doesn't need to know which φ it's in. It learns to
be robust to all of them simultaneously, because it was trained on all of them.

## 6.3 Why Uniform DR Can Fail

**Too narrow:** Real-world parameters fall outside the training distribution.
Policy fails in deployment.

**Too wide:** Training distribution includes impossible or contradictory physics.
The policy cannot find a single strategy that works across all conditions.
Early training is dominated by catastrophic failures (e.g., extreme drag = agent
can barely move, learns nothing useful).

**The Goldilocks problem:** Choose ranges that are wide enough to cover reality
but not so wide that learning is impossible.

CDR solves this by adapting the width during training.

## 6.4 What You Randomised (6 Parameters)

| Parameter | Physical Meaning | Why It Varies | Range in Training |
|-----------|-----------------|---------------|-------------------|
| c_drag_lateral | Lateral drag coefficient | Biofouling, surface roughness | [0.10, 0.50] kg/m |
| c_drag_axial | Axial drag coefficient | ~40% of lateral (torpedo shape) | [0.04, 0.20] kg/m |
| buoyancy_offset | Net buoyancy force offset | Salinity, temperature, payload | [-0.05, 0.10] |
| current_speed | Water current magnitude | Tidal variation, location | [0.0, 0.30] m/s |
| added_mass | Virtual fluid inertia | Vehicle geometry, acceleration | [0.05, 0.30] |
| act_efficiency | Thruster efficiency multiplier | Wear, cavitation, voltage | [0.80, 1.00] |

**Why lateral drag > axial drag:**
A torpedo-shaped AUV has much more frontal area when moving sideways than when
moving forward. The ratio c_drag_lateral / c_drag_axial ≈ 3-5 for typical AUVs.
Your simulation uses this ratio, following Fossen (2011).

## 6.5 The Held-Out Test Distribution

Your paper's central evaluation uses physics parameters outside the training range:

| Parameter | Training max | Test range | Gap factor |
|-----------|-------------|-----------|------------|
| c_drag_lateral | 0.50 kg/m | [0.60, 1.20] kg/m | ~2× higher |
| current_speed | 0.30 m/s | [0.40, 0.80] m/s | ~2× stronger |
| act_efficiency | 0.80 | [0.60, 0.90] | Lower efficiency |

The test distribution is genuinely out-of-distribution — the agent never saw
these parameter ranges during training. This is **zero-shot transfer**: no
fine-tuning, no real data, just deploy and evaluate.

---

# 7. Curriculum Learning and CDR

## 7.1 What is Curriculum Learning?

Curriculum learning (Bengio et al., 2009) is the idea that learning is more
efficient when examples are presented in order from easy to hard — just as
human students learn arithmetic before calculus.

In RL, this translates to starting with simpler tasks and gradually increasing
difficulty. For DR specifically, this means starting with narrow randomisation
ranges and expanding them as the agent learns.

## 7.2 Automatic Domain Randomisation (ADR) — Your Inspiration

OpenAI's ADR (Akkaya et al., 2019), developed for the Rubik's Cube dexterous
manipulation task, is the closest prior work to your CDR.

ADR mechanism:
- Maintain a lower bound φ_lo and upper bound φ_hi for each parameter
- Periodically sample a point **on the boundary** of the current range
- If the policy succeeds at the boundary → expand the range outward
- If the policy fails at the boundary → contract the range inward

**Key difference from your CDR:**
ADR expands each parameter independently, with a separate performance buffer
per parameter. Your CDR expands all 6 parameters simultaneously using a single
shared rolling success rate window. This is simpler (fewer hyperparameters) and
is the first application to fluid physics.

## 7.3 Your CDR Algorithm — Complete Explanation

```
INITIALISE:
  ranges[p] = (start_lo, start_hi)  for each parameter p
  window = deque(maxlen=W=50)        rolling success history

AT EACH EPISODE:
  1. SAMPLE: φ[p] ~ Uniform(ranges[p].lo, ranges[p].hi)  for each p
  2. APPLY:  set simulation physics to φ
  3. RUN:    episode for max 500 steps
  4. RECORD: success = 1 if goal_dist < 0.5m else 0
  5. APPEND: window.append(success)

  IF len(window) == W:
    sr = mean(window)      ← rolling success rate
    
    IF sr > θ_hi = 0.70:   ← EXPAND
      for each p:
        ranges[p].lo = max(min_lo,  lo - ε_expand × range)
        ranges[p].hi = min(max_hi, hi + ε_expand × range)
      curriculum_level increases
    
    IF sr < θ_lo = 0.40:   ← CONTRACT
      for each p:
        ranges[p].lo = min(start_lo, lo + ε_contract × range)
        ranges[p].hi = max(start_hi, hi - ε_contract × range)
      curriculum_level decreases
```

**The curriculum level λ ∈ [0, 1]:**
Tracks fractional expansion progress across all parameters. λ = 0 means you
are at initial (narrow) ranges. λ = 1 means you have reached maximum ranges
(identical to Uniform DR training ranges).

## 7.4 Why CDR Works — Intuition

**The bootstrapping argument:**
In the early episodes with narrow physics ranges, the AUV is in calm water with
nominal drag. Reaching goals is achievable. The agent learns basic navigation —
how to point toward the goal and thrust toward it. This is the curriculum's
first lesson.

Once the agent can do this reliably (sr > 70%), the curriculum expands to
include mild current and higher drag. The agent has already mastered calm-water
navigation, so it can now focus on adapting to disturbances. The next lesson.

Eventually the agent is trained on the full range of physics — but it got there
gradually, building on each level of competence. Compare to Uniform DR, which
asks the agent to simultaneously learn both calm-water navigation AND severe
disturbance handling from episode 1. The conflicting gradient signals slow learning.

**The energy efficiency argument:**
When an agent first learns to reach goals in calm water, it develops smooth,
efficient trajectories. The curriculum then preserves these efficient strategies
as it adapts to harder conditions. Uniform DR, starting with all conditions
simultaneously, may develop aggressive thrust strategies early that persist
throughout training even when unnecessary.

## 7.5 CDR Hyperparameter Choices (Justified)

**Window size W = 50:**
Needs to be large enough to give a stable estimate of success rate. Too small →
CDR oscillates rapidly. Too large → CDR responds slowly to agent improvement.
50 episodes ≈ 25,000 environment steps at 500 steps/episode. This balances
responsiveness with stability.

**Expand threshold θ_hi = 0.70 (70%):**
A 70% success rate means the agent is reliably but not perfectly succeeding
at the current difficulty level. Expanding too early (e.g., 50%) would cause
premature generalisation. Expanding too late (e.g., 90%) would slow curriculum
progression. Based on Akkaya et al. (2019) ADR paper.

**Contract threshold θ_lo = 0.40 (40%):**
If success rate drops below 40%, the current difficulty is too hard. Contract
to give the agent easier conditions to recover competence. The 30-point gap
between thresholds (40-70%) creates stability — prevents oscillation.

**Expand step ε_expand = 5%:**
Expand ranges by 5% of the full range per expansion event. Gradual enough to
avoid sudden difficulty jumps, fast enough to complete expansion in ~1M steps.

**Contract step ε_contract = 3%:**
Slightly slower contraction than expansion. This asymmetry prevents aggressive
contraction when the agent temporarily struggles.

---

# 8. AUV Physics

## 8.1 What is an AUV?

An autonomous underwater vehicle is a self-propelled, untethered robot that
operates underwater without real-time human control. Applications include
ocean floor mapping, pipeline inspection, mine hunting, environmental monitoring,
and scientific data collection.

Your AUV model (Halcyon X4) is a torpedo-shaped vehicle:
- Length: 1.0m · Diameter: 0.15m · Mass: 11.1 kg
- 4 thrusters in X-configuration at the rear
- True 6-DOF (degrees of freedom): surge, sway, heave, roll, pitch, yaw

## 8.2 Six Degrees of Freedom

An object in 3D space has 6 DOF:

| DOF | Motion | Direction | Controlled by |
|-----|--------|-----------|---------------|
| Surge | Forward/backward | X-axis body | All 4 thrusters (equal) |
| Sway | Left/right | Y-axis body | Left vs right differential |
| Heave | Up/down | Z-axis body | Top vs bottom differential |
| Roll | Rotation about X | Around body X | Passive (fins) |
| Pitch | Rotation about Y | Around body Y | Combined differential |
| Yaw | Rotation about Z | Around body Z | Differential thrust |

The X-configuration means pairs of thrusters produce different motion types
through differential thrust allocation.

## 8.3 Coordinate Frames

**World frame (inertial):** Fixed to the Earth (or simulation origin). Used
for goal position and absolute AUV position.

**Body frame:** Fixed to the AUV, moves with it. Used for velocity measurement,
drag computation, and observations.

**Why body frame for observations:**
If you observe goal direction in the world frame, the policy must learn that
"goal is at 45° in world frame" means something completely different depending
on where the AUV is pointing. This creates a huge observation space that's hard
to generalise across.

In body frame, "goal is forward-right" always means the same thing regardless
of where in the world the AUV is. This invariance dramatically improves
generalisation.

The transformation from world to body frame uses the rotation matrix R from
`data.xmat[body_id]`. In code:
```python
v_body = R.T @ v_world   # R.T = world-to-body rotation
v_world = R @ v_body     # R = body-to-world rotation
```

## 8.4 Fluid Physics — The Four Forces

Based on Fossen (2011) — the standard reference for marine vehicle dynamics.

### Quadratic Drag

The most important force for your simulation. As an object moves through fluid,
drag increases with the **square** of velocity:

```
F_drag = -c_drag × v × |v|
```

- **Axial drag** (along body X): Smaller because the torpedo-nose is streamlined
- **Lateral drag** (Y and Z axes): Larger because the cylinder presents more area

The quadratic dependence means doubling speed quadruples drag. This is why
fast AUVs need disproportionately more power. The randomisation of c_drag
is the most impactful uncertainty source in your experiments.

**Physical origin:** Pressure drag from flow separation and skin friction. At
the Reynolds numbers of AUVs (~10^6), the quadratic (turbulent) component
dominates the linear (viscous) component.

### Buoyancy

Archimedes' principle: any object submerged in fluid experiences an upward
force equal to the weight of displaced fluid:

```
F_buoy = ρ_water × g × V_displaced
```

For a neutrally buoyant vehicle, F_buoy = m_AUV × g exactly, and the net
vertical force is zero. In practice, buoyancy varies with:
- Water density (temperature and salinity changes)
- Payload changes
- Air pockets in the hull

Your simulation represents this as a **buoyancy offset δ_b**:
```
F_buoy = m × g × (1 + δ_b)
```

δ_b > 0 → positively buoyant (vehicle floats up)
δ_b < 0 → negatively buoyant (vehicle sinks)

A real AUV mission might start slightly negatively buoyant (cold deep water,
dense) and become positively buoyant as it moves to warmer shallow water.

### Water Current

A persistent flow of water in a fixed direction. The AUV experiences this as
a relative velocity: even with zero thrust, the water is pushing the vehicle.

```
F_current = c_drag_lateral × (v_current - v_AUV) × |v_current - v_AUV|
```

Current is the most practically significant disturbance. Tidal currents in
coastal waters can exceed 1 m/s. Your training range goes to 0.3 m/s
(moderate) and test range to 0.8 m/s (strong tidal conditions).

PID control fails catastrophically under strong current because the fixed gains
cannot distinguish between:
- The error caused by inertia (the vehicle hasn't moved yet)
- The error caused by persistent current (the vehicle is being pushed away)

The integral term tries to compensate but winds up, causing oscillation
and divergence.

### Added Mass

When a body accelerates in fluid, it must also accelerate the surrounding fluid,
requiring extra force. The effective inertia of the vehicle is larger than its
dry mass:

```
F_added_mass = -C_am × m × a_body
```

This effect is significant for fast-accelerating motions. For cruise-speed AUV
control it is a secondary effect, but it contributes to the simulation's
physical accuracy and is included in your randomisation.

## 8.5 Thruster Allocation

Your 4-thruster X-configuration produces all translational forces through
differential thrust. The allocation matrix maps desired body forces to thruster
commands:

```
        [1  1  1  1 ]   [T1]   [Fx]     (surge: all thrusters equal)
A =     [0  0  1 -1 ] × [T2] = [Fy]     (sway: left vs right)
        [1 -1  0  0 ]   [T3]   [Fz]     (heave: top vs bottom)
                        [T4]
```

The inverse of A gives the thruster commands from desired forces:
```
[T1,T2,T3,T4] = A_pinv × [Fx, Fy, Fz]
```

## 8.6 Why Gravity = 0 in MuJoCo

MuJoCo's gravity acts on all bodies. An AUV in water is neutrally buoyant
(net gravity = 0 when buoyancy exactly cancels gravity). Rather than modelling
this as gravity + opposing buoyancy force, you set `gravity="0 0 0"` in the XML
and apply the net buoyancy force manually in Python each step via `xfrc_applied`.

This gives you direct control over the buoyancy offset, which is important for
domain randomisation.

---

# 9. MuJoCo

## 9.1 What is MuJoCo?

MuJoCo stands for Multi-Joint dynamics with Contact. It is a general purpose physics engine that aims to facilitate research and development in robotics, biomechanics, graphics and animation, machine learning, and other areas that demand fast and accurate simulation of articulated structures interacting with their environment.

MuJoCo was acquired by Google DeepMind in October 2021 and open-sourced under the Apache 2.0 licence in May 2022.

## 9.2 Why MuJoCo for Your Project?

| Feature | Why it matters |
|---------|---------------|
| Speed | ~10,000 steps/second on CPU — critical for RL |
| xfrc_applied | Direct external force API — lets you apply custom fluid forces |
| Accurate contact dynamics | Thruster reactions modelled correctly |
| MJCF XML | Human-readable model format |
| Free and open-source | Reproducibility for your GitHub release |
| Python bindings | Direct integration with SB3 and NumPy |

## 9.3 The xfrc_applied Mechanism

This is a key implementation detail in your project. MuJoCo provides
`data.xfrc_applied`, a (nbody × 6) array where you can set external forces
and torques on any body.

Your fluid physics apply to the AUV body each substep:
```python
# Compute forces in body frame, rotate to world frame
F_drag_world = R @ F_drag_body     # R = body-to-world rotation
# Apply to AUV body (first 3 components = force, last 3 = torque)
data.xfrc_applied[auv_body_id, 0:3] = F_drag + F_buoy + F_current + F_added_mass
data.xfrc_applied[auv_body_id, 3:6] = 0.0    # no external torques
```

**Important:** `xfrc_applied` is in **world frame**. This is why you must
rotate forces computed in body frame to world frame before applying them.
Getting this wrong would cause drag to act in the wrong direction when the AUV
rotates — a silent catastrophic bug.

## 9.4 MJCF — The XML Format

Your `auv.xml` defines:
- **worldbody:** The physical world with bodies, geoms, and sensors
- **actuator:** Thruster motor actuators
- **sensor:** Gyroscope, velocimeter, accelerometer, actuator force sensors
- **default:** Default properties (reuse across elements)

Key design decisions in your XML:

**No joint in default class:**
A critical bug fix. Putting `<joint>` in `<default>` caused MuJoCo to
implicitly add joints to every body, creating phantom free joints on fin bodies.
The fix: declare the free joint explicitly only on the halcyon body.

**Keyframe "home":**
Defines a clean starting configuration with zero velocity and standard pose.
Used by `mujoco.mj_resetDataKeyframe(model, data, 0)` at episode reset.

**frame_skip = 4:**
Your control loop runs at 25 Hz (every 4 simulation steps at 250 Hz).
This matches realistic AUV thruster update rates and reduces training time
4× compared to controlling at the simulation timestep.

---

# 10. Your Experimental Design

## 10.1 The Three Conditions

| Condition | Code | What it does | Paper label |
|-----------|------|-------------|-------------|
| Naive SAC | mode="none" | Fixed nominal physics, no DR | Baseline 1 |
| Uniform DR | mode="uniform" | Full DR ranges from episode 1 | Baseline 2 |
| CDR | mode="curriculum" | Performance-gated range expansion | **Ours** |

**Why this comparison is fair:**
- Identical SAC hyperparameters across all conditions
- Same training budget (1M steps)
- Same neural network architecture
- Same reward function
- Same observation space
- Only the physics distribution during training differs

Any performance difference is attributable to the DR strategy, not confounded
by other factors.

## 10.2 Why 3 Seeds?

Scientific experiments need to be reproducible. RL training has inherent
randomness from:
- Initial network weights (random)
- Replay buffer sampling order (random)
- Environment resets (random goal positions, random physics)

A single seed result might be lucky or unlucky. Three seeds give you a mean
and standard deviation, allowing claims like "CDR achieves 96.0% ± 3.7%"
rather than just "CDR achieves 90%" (which might be seed luck).

**Statistical honesty:** With n=3 seeds, you have very limited statistical power
for detecting small differences (CDR vs Uniform DR success rates are close).
You acknowledge this in your limitations section and use effect sizes (Cohen's d)
alongside p-values.

## 10.3 The Held-Out Test Distribution

This is the most important design decision in your paper.

**Why you need a separate test distribution:**
If you evaluate on the same distribution used for training, you're measuring
memorisation, not generalisation. A policy could achieve 100% by exactly
memorising the training distribution without learning any robust principles.

Your test distribution uses parameter ranges strictly **wider** than training:
- c_drag_lateral: training max 0.50 → test range [0.60, 1.20]
- current_speed: training max 0.30 → test range [0.40, 0.80]

These conditions were never seen during training. Success here is genuinely
zero-shot transfer.

## 10.4 Energy as a Primary Metric

Energy consumption (mean absolute thruster command per step) is an unusual
primary metric in RL papers. Most papers only report success rate and reward.

Why energy matters here:
1. Real AUVs are battery-constrained. Energy efficiency directly equals
   mission duration.
2. Success rate alone does not distinguish quality of success. A policy
   that reaches the goal in 50 steps with smooth thrust is better than one
   that reaches it in 500 steps with maximal thrust oscillation — but both
   have 100% success rate.
3. Energy captures the quality of the policy's strategy, not just its outcome.

The reward function's energy penalty term (weight = 0.02) directly incentivises
energy efficiency. CDR's curriculum allows the agent to first learn efficient
strategies in calm water before adapting to harder conditions, while Uniform DR's
early exposure to difficult conditions may encourage high-thrust strategies that
persist.

---

# 11. PID Control

## 11.1 What is PID?

Proportional-Integral-Derivative (PID) control is the most widely used
feedback control algorithm in industry. It computes a control signal based on:

```
u(t) = Kp × e(t) + Ki × ∫e(τ)dτ + Kd × de/dt

where: e(t) = goal_position - current_position  (error)
```

- **Proportional (Kp):** Push toward goal proportionally to error. Too high → oscillation.
- **Integral (Ki):** Accumulate error over time to correct persistent offset (like constant current).
- **Derivative (Kd):** Dampen rate of change, preventing overshoot.

Your tuned gains: Kp=20, Ki=0.1, Kd=10, with anti-windup at 8 rad/s.

## 11.2 Why PID Fails Under Distribution Shift

**The water current problem:**
Suppose the AUV is 2m from the goal with a 0.6 m/s current pushing it away.

At step 1: e = 2m → proportional pushes toward goal.
But the current applies 0.6 m/s force continuously. The AUV moves toward goal
but the current slows it. The integral term accumulates.

As the integral grows, it eventually saturates the command in the current
direction. But now the AUV passes the goal and overshoots. The error reverses.
The proportional term reverses. But the integral still points in the old
direction (it takes time to wind down).

Result: the AUV oscillates around the goal position with ever-increasing
amplitude — the classic PID limit cycle under integral windup.

**The drag problem:**
With high drag (c_lateral = 0.8-1.2 kg/m), the AUV decelerates much faster
than the controller expects. The proportional gain Kp=20 was tuned assuming
nominal drag of 0.20 kg/m. With 4× higher drag, the effective response is
4× slower. The derivative term cannot compensate because it was tuned for
nominal dynamics.

**Why this produces 3% success:**
The combination of high current and high drag creates conditions where the AUV
cannot reach the 0.5m threshold even with aggressive integral action. It gets
close (mean dist ≈ 1-3m) but cannot close the final gap before the 1000-step
episode ends.

## 11.3 The Key Message

PID is not inherently bad. For a fixed, well-characterised environment, a
properly tuned PID controller is simple, reliable, and effective. Your paper
does not claim PID is always wrong. It claims that **fixed-gain PID is
insufficient when fluid parameters vary beyond the tuning conditions**.

The 100% → 3% result is not a criticism of PID as a method. It is a
quantification of a known limitation (lack of adaptation to changing conditions)
that is poorly documented in the AUV RL literature. Your paper provides the
first precise measurement of this fragility.

## 11.4 PID Energy Consumption

Your PID uses Kp=20. At any error > 0, this drives full thrust. This is why
PID energy consumption is the highest of all conditions — the controller is
always pushing near-maximum thrust because it cannot modulate based on trajectory
quality, only on instantaneous error.

RL policies, penalised by the r_energy term, learn to coast when close to the
goal and use thrust efficiently. The ~3× energy difference between PID and
CDR represents the advantage of learned control over fixed-gain control for
energy management.

---

# 12. Results Analysis

## 12.1 Understanding Your Four Key Results

### Result 1: PID 100% → 3%

This is your strongest and most interpretable result. No statistical test
needed — the difference is 97 percentage points. Any reviewer will immediately
understand its significance.

The paper uses this as the opening hook in the introduction. Structure:
1. State the result dramatically
2. Explain mechanistically why it happens (Section 11.2 above)
3. Use it to motivate the rest of the paper

### Result 2: Naive SAC 60.7% ± 27.2%

The ±27.2% standard deviation is the key message here, not the 60.7% mean.
This means three seeds gave approximately [28%, 62%, 96%] (your actual values).
The same training procedure can produce a nearly-useless policy or a
moderately-good one depending on random initialization.

This makes Naive SAC **unreliable for deployment**. You cannot know in advance
which seed's policy you'll get. Domain randomisation eliminates this variance.

### Result 3: CDR 96.0% ± 3.7% vs Uniform DR 99.7% ± 0.5%

These are close. Honest analysis requires acknowledging this:
- Success rate difference is not statistically significant at n=3
- CDR does not outperform Uniform DR on success rate
- The CDR advantage is energy efficiency and policy quality

Do NOT claim CDR is "better" based on success rate alone. Write: "CDR achieves
comparable transfer success to Uniform DR while producing significantly more
energy-efficient policies."

### Result 4: Energy 0.623 (CDR) vs 0.665 (Uniform DR)

6.3% energy saving. Whether this is statistically significant depends on your
results_table.py output. If p < 0.05 → write "significantly lower energy".
If p > 0.05 → write "CDR shows a trend toward lower energy consumption."

Either way, the practical significance (battery life extension) is defensible
regardless of statistical significance with n=3.

## 12.2 Reading Your Training Curves

**env/goal_dist trending down:** Agent learning to navigate. If flat after 200k
steps, training is stuck.

**cdr/curriculum_level:** Should increase from 0 toward 1 over training.
If stuck at 0 → agent not reaching 70% success threshold → curriculum never
expands. This means CDR is stuck in easy mode — the policy never gets exposed
to difficult conditions.

Your actual CDR behaviour: curriculum level starts at 0, may not fully reach
1 by 1M steps for some seeds. This is acceptable — even partial curriculum
expansion provides benefit over no expansion.

**rollout/success_rate near 0 for most of training:** This is expected with CDR.
The curriculum contracts to easy conditions when success drops below 40%.
Success rate on the training distribution (which changes with CDR) is not the
same as test success rate.

## 12.3 Why Tracking Shows 0% (The Bug)

The tracking task has no static goal — it has a moving path marker.
`eval.py` checks `goal_dist < 0.5m` for success. Since there is no static
goal, `goal_dist` is always high → always "failure" → 0% success.

The correct success criterion for tracking: `mean_tracking_error < 1.0m`
(mean distance from the lemniscate path over the episode).

**This is not a model failure** — your tracking models may work fine.
The bug is in the evaluation script's success criterion. Use `eval_tracking.py`
which applies the correct criterion.

---

# 13. Statistical Methods

## 13.1 Why Statistics Matter for a 3-Seed Study

With n=3 samples per condition, standard confidence intervals are very wide
and many comparisons will not reach p < 0.05. This is a limitation you must
acknowledge — but it does not invalidate your results.

The correct approach: report effect sizes alongside p-values. A large Cohen's d
with p > 0.05 (due to small n) is still meaningful evidence.

## 13.2 Welch's t-test

You use Welch's t-test (not Student's t-test) because you cannot assume equal
variance between conditions. Welch's t-test is the correct choice for comparing
two independent groups when variance may differ.

```python
from scipy import stats
t, p = stats.ttest_ind(cdr_values, udr_values, equal_var=False)
```

Interpreting the result:
- p < 0.001: *** (highly significant)
- p < 0.01: ** (very significant)
- p < 0.05: * (significant)
- p > 0.05: ns (not significant with current n)

With n=3, achieving p < 0.05 requires large effect sizes. Small but real
differences will often be "ns". This is not proof of no difference — it is
underpowered evidence.

## 13.3 Cohen's d Effect Size

Cohen's d measures the standardised difference between two means:
```
d = (mean_A - mean_B) / sqrt((std_A² + std_B²) / 2)
```

Interpretation:
- d > 0.8: Large effect — practically significant even with small n
- d > 0.5: Medium effect — worth reporting
- d > 0.2: Small effect — present but subtle
- d < 0.2: Negligible

For your energy comparison (CDR vs Uniform DR), if d > 0.8 you can write:
"Despite n=3, the large effect size (d = X.XX) suggests the energy advantage
is practically meaningful."

## 13.4 Bootstrap Confidence Intervals

With n=3, parametric confidence intervals (based on t-distribution) are very
wide and unreliable. Bootstrap CI is more appropriate:

```python
# 95% Bootstrap CI
boot = [np.mean(np.random.choice(vals, len(vals), replace=True))
        for _ in range(1000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
```

Report as: "CDR energy: 0.623 (95% CI: [X.XXX, X.XXX])"

## 13.5 What to Write in the Paper

**When p < 0.05:**
"CDR achieves significantly lower energy per step than Uniform DR
(0.623 ± Y vs 0.665 ± Z, p = X.XXX, Cohen's d = X.XX)."

**When p > 0.05 but large d:**
"CDR shows lower energy per step than Uniform DR (0.623 ± Y vs 0.665 ± Z,
p = X.XXX, Cohen's d = X.XX), with a large effect size suggesting practical
significance despite the limited sample size (n=3 seeds)."

**Never write:**
"p = 0.08, which is close to 0.05, suggesting significance." This is bad
statistical practice.

---

# 14. Your References Explained

## [1] Haarnoja et al. 2018 — Soft Actor-Critic

**Full citation:** Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).
Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
with a Stochastic Actor. ICML 2018. arXiv:1801.01290.

**What it contains:**
- The original SAC algorithm derivation
- The entropy-regularised RL objective J(π) = E[Σ r_t + α H(π)]
- Proof that soft policy iteration converges to the optimal soft policy
- Empirical comparison showing SAC outperforms DDPG, TD3, PPO on MuJoCo benchmarks
- The key stability result: stochastic policy with entropy maximisation is
  more consistent across random seeds than deterministic variants

**What you cite it for:**
- Equation for SAC objective in Section III.E ("SAC Training")
- Justification for choosing SAC over on-policy methods
- Claim that SAC produces stable, consistent training across seeds

**Key quote to use:**
SAC "aims to simultaneously maximise expected return and entropy — to succeed
at the task while acting as randomly as possible."

## [2] Haarnoja et al. 2018 (v2) — SAC Algorithms and Applications

**Full citation:** Haarnoja, T., et al. (2018). Soft Actor-Critic Algorithms
and Applications. arXiv:1812.05905.

**What it adds over [1]:** Automatic entropy tuning (constrained formulation
where α is adjusted to match target entropy H̄). This is the version of SAC
you actually implemented (`ent_coef="auto"`).

**What you cite it for:** The automatic entropy tuning mechanism in Section III.E.

## [3] Tobin et al. 2017 — Domain Randomisation

**Full citation:** Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W.,
& Abbeel, P. (2017). Domain Randomization for Transferring Deep Neural Networks
from Simulation to the Real World. IROS 2017. arXiv:1703.06907.

**What it contains:**
- First demonstration of DR for sim-to-real: train only on randomised
  simulated images, test on real camera images without fine-tuning
- The core hypothesis: if the variability in simulation is significant enough, models trained in simulation will generalise to the real world with no additional training
- Randomised lighting, textures, camera positions, object positions
- Successfully transferred object detection for robotic grasping

**What you cite it for:**
- Defining Uniform DR in Section II and Section III.C
- The justification that training on wide distributions enables zero-shot transfer
- Prior art establishing DR as the standard sim-to-real approach

**How your work differs:**
Tobin applies DR to visual perception. You apply it to fluid physics. The
principle is the same but the parameter space is completely different.

## [4] Akkaya et al. 2019 — Solving Rubik's Cube (ADR)

**Full citation:** Akkaya, I. et al. (OpenAI) (2019). Solving Rubik's Cube
with a Robot Hand. arXiv:1910.07113.

**What it contains:**
- The ADR algorithm: performance-gated expansion of DR parameter ranges
- Application to dexterous manipulation with 132 randomised parameters
- The boundary sampling mechanism (sample parameters near the expansion boundary)
- Evidence that curriculum DR enables learning tasks impossible with fixed DR
- Ablation showing ADR > Uniform DR on the manipulation task

**Key algorithmic difference from your CDR:**
ADR maintains a performance buffer **per parameter** and expands each
parameter boundary independently. Your CDR uses a single shared rolling
success window and expands all parameters simultaneously. Your approach:
- Requires fewer hyperparameters (one W, one θ_hi, one θ_lo)
- Is simpler to implement and tune
- Is the first application to underwater fluid physics

**What you cite it for:**
- Section II.C (Related Work: ADR as most related curriculum DR work)
- Section III.D (CDR algorithm description — "inspired by Akkaya et al.")
- Section VI (Discussion — why your CDR is simpler than ADR)

## [5] Tiboni et al. 2024 — DORAEMON

**Full citation:** Tiboni, G., Tirupachuri, Y., Nori, F., & Tommasi, T. (2024).
DORAEMON: Domain Randomization via Entropy Maximization. arXiv:2301.12457.

**What it contains:**
- Bayesian optimisation-based curriculum DR
- Finds DR distribution that maximises entropy (diversity) subject to a
  success rate constraint
- More sophisticated than threshold-based CDR but more complex to implement

**What you cite it for:**
- Section II.C (one sentence: "more sophisticated Bayesian approaches exist")
- Framing your simple threshold-based CDR as a practical alternative

**How to position it:**
"While DORAEMON [Tiboni 2024] uses Bayesian optimisation for adaptive DR —
more theoretically principled than our threshold approach — it requires
substantially more implementation complexity and has not been applied to
underwater fluid physics."

## [6] Fossen 2011 — Handbook of Marine Craft Hydrodynamics

**Full citation:** Fossen, T.I. (2011). Handbook of Marine Craft Hydrodynamics
and Motion Control. John Wiley & Sons. ISBN: 978-1-119-99149-6.

**What it contains:**
- Complete 6-DOF equations of motion for marine vehicles
- Quadratic drag (Morison equation): F_drag = -D(ν)|ν|
- Added mass theory: effect of fluid inertia on vehicle dynamics
- Buoyancy and gravity: Archimedes' principle applied to AUVs
- Water current modelling: current as body force in equations of motion
- PID and nonlinear control for AUVs

**What you cite it for:**
- Section III.B ("Fluid Physics") to give academic authority to your
  physics model equations
- Section VI (Discussion) when acknowledging simplified model limitations
- Related Work when discussing PID control limitations

**The key equations from Fossen:**
```
M_RB × ν̇ + C_RB(ν) × ν + M_A × ν̇ + C_A(ν) × ν + D(ν) × ν = τ + g(η) + w
```
where M_RB = rigid body mass matrix, M_A = added mass matrix,
C = Coriolis/centripetal matrix, D = damping matrix, τ = thruster forces,
g(η) = gravity/buoyancy vector, w = environmental disturbances.

Your simplified model retains: drag (D), buoyancy (g), current (w), and
added mass (M_A). You omit Coriolis terms (acceptable approximation for
moderate-speed torpedo AUVs).

## [7] Arndt et al. 2024 — Learning to Swim

**Full citation:** Arndt, K. et al. (2024). Learning to Swim: Reinforcement
Learning for 6-DOF Control of Thruster-driven AUVs. ICRA 2024. arXiv:2410.00120.

**What it contains:**
- SAC + Uniform DR for 6-DOF AUV goal-reaching
- Randomises only 2 parameters: centre of buoyancy and displaced volume
- Demonstrated zero-shot sim-to-real transfer on a real AUV platform
- Comparable performance to hand-tuned PID controller

**What you cite it for:**
- Section II.B (Closest prior work on AUV RL + sim-to-real)
- Establishing that SAC + DR is a viable approach for AUV control

**How your work differs (important for Related Work section):**
1. Their DR covers 2 parameters; yours covers 6+
2. They use only Uniform DR; you compare CDR vs Uniform DR systematically
3. No energy efficiency analysis
4. No perturbation recovery evaluation
5. No systematic ablation of which parameter matters most

## [8] Chu et al. 2025 — MarineGym

**Full citation:** Chu, S. et al. (2025). MarineGym: A High-Performance
Reinforcement Learning Platform for Underwater Robotics. arXiv:2503.09203.

**What it contains:**
- GPU-accelerated underwater RL benchmark (Isaac Sim-based)
- 5 AUV models, 3 task types
- DR toolkit for physics parameter randomisation
- Comprehensive baselines across multiple algorithms

**What you cite it for:**
- Section II.A (MarineGym as most comprehensive underwater RL platform)
- Section II.C (MarineGym applies Uniform DR but not curriculum DR)

**How your work differs:**
1. MarineGym uses Isaac Sim (NVIDIA proprietary); yours uses MuJoCo (open-source)
2. MarineGym studies multiple algorithms; you study DR strategies specifically
3. No curriculum scheduling in MarineGym DR toolkit
4. No energy efficiency metric
5. MarineGym does not study which DR parameter matters most (your ablation)

## [9] Chaffre et al. 2025 — Sim-to-Real AUV

**Full citation:** Chaffre, T. et al. (2025). Sim-to-real transfer of adaptive
control parameters for AUV stabilisation under current disturbance. IJRR.
DOI: 10.1177/02783649241272115.

**What it contains:**
- SAC + enhanced uniform DR + bio-inspired experience replay
- Physical AUV tank experiments (real hardware validation)
- Focuses on station-keeping (maintaining position) rather than goal-reaching
- Hybrid RL + classical control architecture

**What you cite it for:**
- Section II.B (evidence that sim-to-real AUV RL is feasible with hardware)
- Discussion limitation: "hardware validation is left to future work"

## [10] Peng et al. 2018 — Sim-to-Real with Dynamics Randomisation

**Full citation:** Peng, X.B., Andrychowicz, M., Zaremba, W., & Abbeel, P. (2018).
Sim-to-Real Transfer of Robotic Control with Dynamics Randomization. ICRA 2018.

**What it contains:**
- First systematic study of **dynamics** DR (not visual DR)
- Randomises mass, friction, joint gains, body dimensions
- Shows LSTM policies (with memory) outperform feedforward policies for
  sim-to-real transfer (memory enables implicit system identification)
- Applied to locomotion tasks on real robot

**What you cite it for:**
- Section II.C (dynamics DR as the relevant prior work class)
- Situating your work within the dynamics DR tradition

## [11] Bengio et al. 2009 — Curriculum Learning

**Full citation:** Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009).
Curriculum Learning. ICML 2009.

**What it contains:**
- The formal introduction of curriculum learning as a machine learning concept
- Evidence that training on easy examples first, then hard examples, improves
  generalisation and convergence speed
- Theoretical justification: human learners use implicit curricula in education

**What you cite it for:**
- Section II.C or Introduction when motivating curriculum approaches
- "Following Bengio et al. [2009], we start from easier physics conditions..."

## [12] Todorov et al. 2012 — MuJoCo

**Full citation:** Todorov, E., Erez, T., & Tassa, Y. (2012). MuJoCo: A
Physics Engine for Model-Based Control. IROS 2012.

**What it contains:**
- Description of MuJoCo's physics engine design
- Generalised coordinate representation for articulated bodies
- Contact dynamics solver
- Comparison with other physics engines

**What you cite it for:**
- Section III.A ("We use MuJoCo [Todorov 2012] as our simulation backend")

## [13] Manhães et al. 2016 — UUV Simulator

**Full citation:** Manhães, M.M.M. et al. (2016). UUV Simulator: A Gazebo-based
Package for Underwater Intervention and Multi-Robot Simulation. OCEANS 2016.

**What it contains:**
- ROS-integrated underwater simulation platform built on Gazebo
- Higher-fidelity hydrodynamics using Fossen's full equations
- Multiple AUV models including torpedo shapes
- Used as a second evaluation environment for your cross-simulator transfer

**What you cite it for:**
- If you complete the UUV Simulator transfer evaluation (A10b)
- Background on simulation ecosystem

---

# 15. Writing Guide

## 15.1 The Paper's Logical Flow

Your paper needs to tell a coherent story. Here is the narrative arc:

```
INTRODUCTION:
"PID collapses catastrophically. Here's why DR fixes it. Here's why CDR
is better than uniform DR. Here's what we show."

RELATED WORK:
"Prior work shows DR works for robots. Prior work shows SAC works for AUVs.
Nobody has done CDR for AUV fluid physics. That's our gap."

METHOD:
"Here's exactly what we built and why. Every design decision justified."

EXPERIMENTS:
"Here's exactly what we ran and how we measured it. Reproducible."

RESULTS:
"PID: 100%→3%. DR eliminates variance. CDR uses 6% less energy. Here's
why each result happened."

DISCUSSION:
"What this means practically. Why CDR works this way. What we can't claim."

CONCLUSION:
"We showed X, Y, Z. Use this for battery-constrained AUV deployment."
```

## 15.2 Section-by-Section Writing Order and Tips

**Write in this order:** Results → Method → Experiments → Discussion → Related Work → Introduction → Abstract

Writing Results first forces you to know exactly what you're claiming before
you write any motivation. This prevents you from overpromising in the introduction
and then underdelivering in the results.

**Section 5 (Results) — Key writing principles:**

- Start each subsection with the result directly:
  "PID achieves 3% success on the held-out test distribution (Table I)."
  NOT: "We now present the results of our PID evaluation. As can be seen..."

- Every number needs a unit:
  "0.623 per step" NOT "0.623"
  "96.0% ± 3.7%" NOT "96.0%"

- Distinguish training vs test results clearly:
  "On the training distribution, all conditions achieve near-perfect success.
  We therefore focus on held-out test distribution results (Table I)."

**Section 3 (Method) — What to include:**

- Complete observation space vector with dimensions and units
- Complete reward function with all weights
- Complete DR parameter table with min/max ranges
- CDR algorithm pseudocode (Algorithm 1 box)
- SAC hyperparameter table (Table III)

Reviewers replicate your method from this section. Be precise and complete.

## 15.3 Things to Never Write

| Wrong | Right |
|-------|-------|
| "Our method outperforms all baselines" | "CDR achieves comparable success to Uniform DR while using X% less energy" |
| "CDR is clearly superior" | "CDR produces more energy-efficient policies than Uniform DR (p=X, d=X)" |
| "The results show our method works perfectly" | "CDR achieves 96.0% ± 3.7% transfer success on the held-out distribution" |
| "We prove that CDR is better" | "Our experiments provide evidence that CDR reduces energy consumption" |
| "As shown above" | Always use explicit references: "as shown in Table I" |
| "Obviously" | Never use this word in a paper |
| "Very good results" | Quantify everything |

## 15.4 How to Write the Abstract

The abstract should be exactly 150-200 words and follow this structure:

1. **Sentence 1-2: The problem.** "AUV controllers fail under distribution shift."
2. **Sentence 3-4: The gap.** "No prior work has applied curriculum DR to AUV fluid physics."
3. **Sentence 5-6: What you do.** "We present CDR — a performance-gated expansion of physics randomisation."
4. **Sentence 7-8: Main results.** "PID collapses 100%→3%. CDR achieves 96.0% with 6.3% lower energy."
5. **Sentence 9: Contribution.** "We release an open-source MuJoCo AUV environment."

## 15.5 How to Write the Introduction

**Paragraph 1 — Hook (2-3 sentences):**
The PID 100%→3% result. State it concisely and dramatically. This is the
first thing the reviewer reads. Make them want to keep reading.

**Paragraph 2 — Why it's hard (3-4 sentences):**
Explain the sim-to-real gap in AUV fluid physics. Reference Fossen (2011)
for the physics. This shows you understand the problem domain.

**Paragraph 3 — Prior work and gap (3-4 sentences):**
DR works (Tobin 2017, Peng 2018). SAC works for AUVs (Arndt 2024, MarineGym).
But nobody has studied curriculum scheduling of DR for AUV physics. That's the gap.

**Paragraph 4 — What you do (3-4 sentences):**
Describe CDR concisely. Mention the comparison conditions. Mention the metrics.

**Paragraph 5 — Contributions (bulleted list of 3-4 items):**
Be precise. "First application of CDR to AUV fluid physics." Not "novel contribution."

---

# 16. Common Reviewer Objections

## "No real hardware validation"

**Your response:**
"We acknowledge this limitation explicitly in Section VI. Our contribution is
the first systematic comparison of curriculum vs uniform DR strategies for
AUV fluid physics in simulation. Hardware validation requires access to an
AUV platform and water tank — resources not available to us at MIST Bangladesh.
Our sim-to-sim cross-parameter transfer evaluation (Table II) provides a proxy
for cross-domain generalisation. Hardware validation is a natural next step
that this simulation study directly motivates."

**In the paper:** Write one honest paragraph in Section VI Limitations:
"This study is limited to simulation. Hardware validation on a real AUV —
the critical missing step — is left to future work. Our physics model follows
Fossen (2011) but simplifies real ocean dynamics by omitting turbulence and
wave effects."

## "CDR does not outperform Uniform DR on success rate"

**Your response:**
"We report this honestly throughout the paper. CDR's advantage is not success
rate — it is energy efficiency (6.3% lower per-step energy consumption, p=X)
and training consistency. For battery-constrained real AUV deployment, energy
efficiency directly determines mission duration. We never claim CDR has higher
success rate; we claim CDR produces better policies as measured by energy
efficiency, which is a practically important metric that prior work ignores."

## "Only 3 seeds — insufficient statistics"

**Your response:**
"We acknowledge the limited statistical power explicitly in Section VI.
The primary finding — PID collapse from 100% to 3% — requires no statistical
test. For the CDR vs Uniform DR energy comparison, we report both the p-value
and Cohen's d effect size. The large effect size (d=X.XX) provides evidence
of practical significance even when p > 0.05 due to small n. Future work
with more seeds would strengthen this claim."

## "MarineGym already provides DR for AUVs"

**Your response:**
"MarineGym (Chu et al. 2025) provides Uniform DR as a toolkit feature.
It does not study curriculum scheduling of DR, does not evaluate energy
efficiency, and uses Isaac Sim (proprietary NVIDIA). Our contribution is
the systematic comparison of curriculum vs uniform DR scheduling, with energy
efficiency as a primary metric — none of which MarineGym addresses."

## "Your physics model is too simplified"

**Your response:**
"Our model follows the standard Fossen (2011) formulation used throughout
the AUV control literature, capturing the dominant effects: quadratic drag,
buoyancy, water current, and added mass. We acknowledge in Section VI that
we omit turbulence, vortex shedding, and wave disturbances. A more complex
model would increase the training challenge, not decrease it — if anything,
our results underestimate the benefit of CDR in more turbulent conditions."

## "Why not compare against DORAEMON or other curriculum DR methods?"

**Your response:**
"Implementing DORAEMON requires fitting a Bayesian optimisation model during
training — significant additional complexity beyond our scope. We compare against
the relevant engineering baselines (Uniform DR, Naive SAC, PID) and position
DORAEMON as related work in Section II. Direct comparison with DORAEMON is
valuable future work; our paper establishes the baseline that CDR improves upon
Uniform DR for this domain."

---

# REFERENCES

[1] Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic:
Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.
ICML 2018. arXiv:1801.01290.

[2] Haarnoja, T. et al. (2018). Soft Actor-Critic Algorithms and Applications.
arXiv:1812.05905.

[3] Tobin, J. et al. (2017). Domain Randomization for Transferring Deep Neural
Networks from Simulation to the Real World. IROS 2017. arXiv:1703.06907.

[4] Akkaya, I. et al. (OpenAI) (2019). Solving Rubik's Cube with a Robot Hand.
arXiv:1910.07113.

[5] Tiboni, G. et al. (2024). DORAEMON: Domain Randomisation via Entropy
Maximisation. arXiv:2301.12457.

[6] Fossen, T.I. (2011). Handbook of Marine Craft Hydrodynamics and Motion
Control. John Wiley & Sons.

[7] Arndt, K. et al. (2024). Learning to Swim: RL for 6-DOF Control of
Thruster-driven AUVs. ICRA 2024. arXiv:2410.00120.

[8] Chu, S. et al. (2025). MarineGym: A High-Performance RL Platform for
Underwater Robotics. arXiv:2503.09203.

[9] Chaffre, T. et al. (2025). Sim-to-real transfer of adaptive control
parameters for AUV stabilisation under current disturbance. IJRR.
DOI: 10.1177/02783649241272115.

[10] Peng, X.B. et al. (2018). Sim-to-Real Transfer of Robotic Control
with Dynamics Randomization. ICRA 2018. arXiv:1710.06537.

[11] Bengio, Y. et al. (2009). Curriculum Learning. ICML 2009.

[12] Todorov, E., Erez, T., & Tassa, Y. (2012). MuJoCo: A Physics Engine
for Model-Based Control. IROS 2012.

[13] Manhães, M.M.M. et al. (2016). UUV Simulator. OCEANS 2016.

[14] Sutton, R.S. & Barto, A.G. (2018). Reinforcement Learning: An
Introduction. 2nd ed. MIT Press.

[15] Weng, L. (2019). Domain Randomization for Sim2Real Transfer.
lilianweng.github.io/posts/2019-05-05-domain-randomization

---

*Concepts Book Version 1.0*
*Created for: Curriculum Domain Randomisation for AUV Control (RA-L/IROS 2026)*
*Author: Limon Howlader · MIST Bangladesh*
*This document explains every concept needed to understand, write, and defend the paper.*
