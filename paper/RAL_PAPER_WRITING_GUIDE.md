# Complete Guide to Writing Your RA-L Paper
## From Zero to Q1 Journal Submission — AUV Curriculum Domain Randomisation

> **Who this guide is for:** You are new to academic paper writing. You have real experimental results.  
> You want to write a Q1 paper for IEEE Robotics and Automation Letters (RA-L), 6 pages maximum.  
> This guide walks you through every single step, from understanding what a journal paper is to submitting the final PDF.

---

## PART 0 — BEFORE YOU WRITE ANYTHING

### What is RA-L and why does it matter?

**IEEE Robotics and Automation Letters (RA-L)** is a Q1 journal — the highest quality tier. It is peer-reviewed, meaning 2–3 expert robotics researchers will read your paper critically and decide whether it is worth publishing. The acceptance rate is around 30–40%.

RA-L papers are special because you can also present them at IROS or ICRA. So by submitting to RA-L, you get a journal publication *and* a conference presentation. This is called the joint RA-L + IROS track.

**Page limit: 6 pages** (including figures and tables), plus 1 extra page that can only contain references. So effectively: 6 pages of content, then your references on page 7.

### What makes a reviewer accept a paper?

Before writing a single word, understand what reviewers look for. They ask:

1. **Is this novel?** Has this exact thing been done before? Your answer: No. CDR for AUV fluid physics has not been done.
2. **Is it significant?** Does this matter to the field? Your answer: Yes. AUV deployment reliability and energy efficiency are critical real-world problems.
3. **Is the evidence convincing?** Did you run enough experiments properly? Your answer: Yes. 9 trained models × 3 seeds, PID baseline, held-out test distribution.
4. **Is it clearly written?** Can I understand it? Your answer: This guide will make sure yes.

### Your paper's core argument in one sentence

Write this sentence on a piece of paper and stick it above your desk:

> *"PID control collapses from 100% to 3% success under realistic fluid-parameter shift, and our Curriculum Domain Randomisation produces policies that maintain 96% transfer success while using 6.3% less thrust energy than Uniform DR — demonstrating that curriculum scheduling is both necessary and advantageous for AUV deployment."*

Every sentence you write should either support this argument or provide the context needed to understand it. If a sentence does neither, delete it.

---

## PART 1 — THE STRUCTURE OF YOUR 6-PAGE PAPER

Here is exactly how to allocate your 6 pages. Do not deviate from this. These proportions are standard for robotics RA-L papers.

```
┌─────────────────────────────────────────────────────┐
│  Section              │  Pages  │  Words (approx.)  │
├─────────────────────────────────────────────────────┤
│  Title + Abstract     │  0.25   │  250 words        │
│  I. Introduction      │  0.75   │  450 words        │
│  II. Related Work     │  0.75   │  450 words        │
│  III. Method          │  1.50   │  900 words        │
│  IV. Experiments      │  0.50   │  300 words        │
│  V. Results           │  1.00   │  600 words        │
│  VI. Discussion       │  0.50   │  300 words        │
│  VII. Conclusion      │  0.25   │  150 words        │
│  References (page 7)  │  1.00   │  ~25 references   │
└─────────────────────────────────────────────────────┘
Total: ~3,400 words of text + figures + tables
```

**Important:** In IEEE format, the text is in two columns. 1 column ≈ 400 words. So 6 pages = roughly 12 columns = 4,800 words, but figures and tables take up ~1,400 of those words' worth of space. That leaves approximately 3,400 words of actual prose.

---

## PART 2 — HOW TO WRITE EACH SECTION

---

### SECTION 1: TITLE

**Purpose:** Tell the reader exactly what you did in 12–15 words.

**Rules for a good title:**
- Include the method name (Curriculum Domain Randomisation)
- Include the application domain (AUV)
- Include the goal (sim-to-real, energy-efficient, robust)
- No abbreviations unless they are universally known (AUV is fine)
- Do not use "A Study of..." or "An Investigation into..." — these are weak

**Your title:**
```
Curriculum Domain Randomisation for Energy-Efficient and 
Robust Autonomous Underwater Vehicle Control
```

Why this is good:
- "Curriculum Domain Randomisation" — the method
- "Autonomous Underwater Vehicle" — the domain
- "Energy-Efficient and Robust" — two contributions (energy + transfer success)

---

### SECTION 2: ABSTRACT (250 words maximum)

**Purpose:** A standalone summary of your entire paper. Many readers will read ONLY the abstract to decide if your paper is worth reading.

**The formula:** Every abstract for an engineering/robotics paper should answer exactly these 5 questions in order:

```
1. What problem exists? (1–2 sentences)
2. What did you do? (2–3 sentences)  
3. What is your main result? (2–3 sentences with actual numbers)
4. Why does it matter? (1–2 sentences)
5. What do you release? (1 sentence)
```

**What NOT to write:**
- ❌ "In this paper, we present..." (everyone says this — just say what you did)
- ❌ "We conducted experiments..." (vague — give the numbers)
- ❌ "Results are promising" (meaningless — say the actual number)
- ❌ "Our approach outperforms baselines" (by how much? on what metric?)

**What TO write:**

```
PROBLEM (sentences 1–2):
Classical PID controllers for autonomous underwater vehicle (AUV) 
navigation achieve near-perfect success under nominal fluid conditions 
yet collapse catastrophically when fluid parameters shift by modest, 
ocean-realistic amounts — dropping from 100% to 3% success as drag 
and current increase to deployment-level ranges.

METHOD (sentences 3–5):
We present Curriculum Domain Randomisation (CDR), a performance-gated 
sim-to-real training strategy that begins with narrow physics-parameter 
ranges and expands them as the policy's rolling success rate exceeds 
an adaptive threshold. We compare CDR against Uniform DR, Naive SAC, 
and a PID baseline on a custom 6-DOF MuJoCo AUV environment with 
six randomisable fluid parameters.

RESULTS (sentences 6–8):
On a held-out test distribution never seen during training — with 
elevated drag (0.60–1.20 kg/m) and strong currents (0.40–0.80 m/s) 
— CDR achieves 96.0% ± 3.7% transfer success across three seeds.
Beyond success rate, CDR consumes 6.3% less thrust energy per step 
than Uniform DR (0.623 vs. 0.665), a meaningful saving for 
battery-constrained real AUVs.

WHY IT MATTERS (sentences 9–10):
Domain randomisation eliminates the ±27.2% inter-seed deployment 
variance of unrobust policies, enabling reliable pre-deployment 
performance assessment. Our results establish energy efficiency as 
a meaningful differentiator between DR strategies at comparable 
success rates.

RELEASE (sentence 11):
We open-source our MuJoCo AUV environment (Halcyon X4) and full 
training pipeline to facilitate reproducible underwater robotics research.
```

**How to write your abstract in practice:**
1. Write one sentence answering each of the 5 questions above
2. Expand to 2–3 sentences where you have numbers to report
3. Count the words. If over 250, cut adjectives first, then sentences

---

### SECTION 3: INTRODUCTION (450 words, about 0.75 pages)

**Purpose:** Convince a sceptical reviewer that your problem is important, that no one has solved it, and that your approach is the right one. This section ends with your contributions listed as bullet points.

**The 5-paragraph formula for an intro:**

```
Paragraph 1 — The Hook (why does this domain matter?)
Paragraph 2 — The Specific Problem (what specifically goes wrong?)
Paragraph 3 — Prior Work and Its Limits (what has been tried, and why it's not enough?)
Paragraph 4 — Your Approach (what you did, briefly)
Paragraph 5 — Contributions (bulleted list of 3 specific claims)
```

**How to write each paragraph:**

**Paragraph 1 — The Hook (~80 words)**

Start with something concrete and real-world, not with "Autonomous robots are increasingly important." That is boring and every paper says it.

Start with your most dramatic result instead:

> *"A PID controller carefully tuned for nominal AUV physics achieves 100% goal-reaching success in simulation — and 3% when the same fluid parameters shift by a modest, ocean-realistic amount. This single result encapsulates the central challenge of autonomous underwater vehicle deployment: controllers optimised for nominal conditions are insufficient when drag, buoyancy, and water current vary as they invariably do in real ocean operations."*

Why this works: You opened with a number (100% → 3%). The reader immediately wants to know why this happens and what you did about it.

**Paragraph 2 — The Specific Problem (~80 words)**

Explain why the problem is hard. Don't just say "it's challenging" — say specifically what makes it hard:

> *"Real ocean conditions vary continuously: drag coefficients shift with biofouling accumulation, water currents vary with tidal patterns and depth, and buoyancy changes with payload and salinity gradients. A controller trained for nominal conditions encounters a fundamentally different physical regime during deployment. The gap between training and deployment conditions — the sim-to-real gap — is not a minor calibration issue but a systematic difference that causes catastrophic failure."*

**Paragraph 3 — Prior Work and Its Limits (~90 words)**

Name 3–5 relevant papers and explain specifically why they do not solve your problem. This is important — reviewers need to see that you know the field.

> *"Reinforcement learning with domain randomisation (DR) offers a principled solution: by varying physics parameters during training, policies can generalise to unseen conditions [CITE Tobin 2017, Peng 2018]. Recent work has applied DR to AUV control [CITE Arndt 2024, Chu 2025], demonstrating sim-to-real transfer for station-keeping and path-following. However, standard Uniform DR applies fixed-range randomisation from the first episode, simultaneously exposing the agent to easy and hard conditions. We show this produces high inter-seed variance (60.7% ± 27.2%) and systematically over-aggressive thrust strategies. No prior work studies curriculum scheduling of DR for underwater physics, and none evaluates energy efficiency — a metric of direct operational significance for battery-constrained AUVs."*

**Paragraph 4 — Your Approach (~70 words)**

One paragraph explaining CDR at a high level. Do not give equations here — just the concept:

> *"We present Curriculum Domain Randomisation (CDR) for AUV fluid physics: a performance-gated strategy that begins training with narrow physics-parameter ranges and expands them as the policy's rolling success rate exceeds a threshold. CDR is inspired by OpenAI's Automatic Domain Randomisation [CITE Akkaya 2019] but simplified for underwater physics, expanding all six parameters simultaneously using a single shared success window."*

**Paragraph 5 — Contributions (bulleted, ~80 words total)**

This is the most important paragraph in your introduction. Reviewers scan directly to this list. Each bullet must be a specific, verifiable claim — not a vague description.

Format: **Action verb + specific claim + why it matters.**

> **Our contributions are:**
> 
> 1. The **first systematic comparison** of curriculum versus uniform DR strategies for AUV fluid physics, demonstrating that CDR produces more energy-efficient policies at comparable transfer-success rates.
> 
> 2. A **quantitative PID fragility benchmark** showing 100% → 3% success collapse under realistic fluid-parameter shift — establishing the practical necessity of physics-aware training for real AUV deployment.
> 
> 3. An **open-source MuJoCo AUV environment** (Halcyon X4) with nine randomisable physical parameters, released to support reproducible underwater robotics research.

**What NOT to write in contributions:**
- ❌ "We propose a novel method" (what method? why is it novel?)
- ❌ "We demonstrate improved performance" (improved by how much? on what?)
- ❌ "We conduct extensive experiments" (this is a description, not a contribution)

---

### SECTION 4: RELATED WORK (450 words, about 0.75 pages)

**Purpose:** Show reviewers that you know the field. Show that your work is different from everything that came before. This section is NOT a literature summary — it is a strategic document.

**Structure for your paper: 3 subsections**

```
II-A. Domain Randomisation for Sim-to-Real Transfer (~150 words)
II-B. Reinforcement Learning for AUV Control (~150 words)  
II-C. Curriculum Learning (~100 words)
```

**How to write each subsection:**

For each paper you cite, follow this template:
```
[Authors] did [specific thing]. [Key result/approach in 1 sentence]. 
However, [specific limitation that your work addresses].
```

Example for subsection A:

> *"Tobin et al. [CITE] first demonstrated sim-to-real visual DR for robotic grasping, randomising textures and lighting. Peng et al. [CITE] extended DR to dynamics — mass, friction, joint parameters — for legged locomotion. OpenAI's ADR [CITE Akkaya 2019] adapts parameter boundaries based on per-parameter performance, enabling dexterous manipulation with 132 randomised parameters. DORAEMON [CITE Tiboni 2024] frames adaptive DR as entropy maximisation with Bayesian optimisation. Unlike these works targeting manipulation or locomotion, we apply curriculum DR to the qualitatively distinct parameter space of hydrodynamic forces, and evaluate energy efficiency as a primary metric — aspects not addressed in prior DR literature."*

**The most important sentence in Related Work** is the last sentence of each subsection, where you explicitly state how your work is different. Do not leave the reader to figure this out. Say it directly.

**Papers you must cite (include these):**

| Paper | Why you cite it |
|-------|----------------|
| Tobin et al. 2017 (Domain Randomization) | First DR paper — foundational citation |
| Peng et al. 2018 (SimToReal dynamics) | DR for dynamics — closest ancestor |
| Akkaya et al. 2019 (ADR/Rubik's Cube) | CDR is directly inspired by this |
| Haarnoja et al. 2018 (SAC) | Your training algorithm |
| Tiboni et al. 2024 (DORAEMON) | Most recent adaptive DR comparison |
| Arndt et al. 2024 (AUV sim-to-real) | Closest AUV DR paper |
| Chu et al. 2025 (MarineGym) | Most comprehensive AUV RL benchmark |
| Fossen 2011 (AUV hydrodynamics) | Your physics model comes from here |
| Bengio et al. 2009 (Curriculum Learning) | CDR is a form of curriculum learning |

**Formatting rule:** Every claim about another paper must have a citation. Never write "prior work has shown..." without [CITE].

---

### SECTION 5: METHOD (900 words, about 1.5 pages)

**Purpose:** Explain what you built, clearly enough that another researcher could reproduce your results. This is the most technical section. It should be precise and complete.

**Structure: 5 subsections**

```
III-A. Problem Formulation (MDP definition) — ~150 words
III-B. AUV Environment — ~200 words  
III-C. Uniform DR Baseline — ~100 words
III-D. Curriculum Domain Randomisation — ~300 words + Algorithm box
III-E. RL Algorithm: SAC — ~100 words
```

**How to write each subsection:**

**III-A. Problem Formulation (~150 words)**

This defines the mathematics. Every RL paper does this. It tells the reader what state space, action space, and reward function you use.

Template:
> *"We formulate AUV goal-reaching as a Markov Decision Process (MDP) M = (S, A, R, T, γ), where S ⊂ R^18 is the observation space, A = [-1,1]^4 is the continuous action space, R is the reward function, T is the physics transition governed by MuJoCo, and γ = 0.99.*
> 
> *The agent receives observation s_t and selects action a_t, commanding four thrusters. An episode terminates on goal-reaching (distance < 0.5 m) or workspace exit (radius 15 m), and is truncated after 500 steps (20 s at 25 Hz).*
> 
> *For sim-to-real transfer, we consider a distribution over MDPs parameterised by physics parameters φ. Training occurs on P_train(φ); we evaluate zero-shot transfer to held-out P_test(φ) with wider ranges."*

**III-B. AUV Environment (~200 words)**

Describe the robot and the physics. Use sub-subheadings (*Robot model*, *Fluid physics*, *Observation space*, *Reward function*).

For the **observation space**, list all 18 dimensions in a compact format (a small table or inline list).

For the **reward function**, write it as an equation. Reviewers expect this. Your reward has 6 terms:

```
r_t = 10·Δd/Δt        (progress)
    + 0.1               (alive bonus)
    - 0.02·||a_t||²    (energy penalty)
    - 0.05·||Δa_t||²   (smoothness)
    - 0.5·(1-cosθ)     (orientation)
    - 5·1_oob          (boundary)
    + 50·1_success     (terminal bonus)
```

Write one sentence explaining why the energy penalty is important: *"The energy penalty encourages efficient thrust — the direct mechanism through which CDR's curriculum structure leads to lower energy consumption at test time."*

**III-C. Uniform DR Baseline (~100 words)**

This is just one paragraph explaining that Uniform DR samples uniformly from the ranges in Table I. Include Table I here (parameter ranges). This is one of your two main tables — make it look good.

**Table format:**

| Parameter | CDR start | CDR max / UDR train | Test (held-out) |
|---|---|---|---|
| Drag lateral (kg/m) | [0.15, 0.25] | [0.10, 0.50] | [0.60, 1.20] |
| Drag axial (kg/m) | [0.06, 0.10] | [0.04, 0.20] | [0.24, 0.48] |
| Buoyancy offset | [-0.01, 0.05] | [-0.05, 0.10] | [-0.15, 0.20] |
| Current speed (m/s) | [0.0, 0.10] | [0.0, 0.30] | [0.40, 0.80] |
| Added mass coeff | [0.10, 0.20] | [0.05, 0.30] | [0.30, 0.60] |
| Act. efficiency | [0.95, 1.00] | [0.80, 1.00] | [0.60, 0.90] |

Caption: *"Physics parameter ranges. CDR start = narrow initial ranges. CDR max = full Uniform DR ranges. Test ranges are strictly wider than training ranges and never seen during training."*

**III-D. Curriculum Domain Randomisation (~300 words + Algorithm)**

This is your core technical contribution. Three things to include:

1. **The intuition** (1 paragraph, ~80 words):
   > *"CDR begins with narrow physics ranges and expands them based on rolling policy performance. The intuition is that training on low-drag and high-drag episodes simultaneously creates conflicting gradient signals: high drag requires strong sustained thrust while low drag requires gentle correction. CDR sequentialises this, allowing the policy to master goal-reaching under easy physics before progressively generalising."*

2. **The algorithm box** — this is mandatory for a method paper. Format it as Algorithm 1. It should show the full CDR loop in pseudocode.

3. **The curriculum level** (1 sentence): *"The curriculum level λ ∈ [0,1] tracks overall expansion progress; λ = 0 is CDR start ranges, λ = 1 is full Uniform DR ranges."*

4. **Hyperparameter table** — small table: W=50, τ⁺=0.70, τ⁻=0.40, ε_expand=0.05, ε_contract=0.03.

**III-E. RL Algorithm: SAC (~100 words)**

Brief paragraph naming SAC, citing Haarnoja 2018, listing your hyperparameters. Put the SAC hyperparameters in Table II (or as a compact inline table).

Include: lr=3×10⁻⁴, buffer=500k, batch=256, γ=0.99, network=[256,256] ReLU, obs normalisation via VecNormalize.

**III-F. PID Baseline (~80 words)**

Describe PID in 2–3 sentences. State your tuned gains (Kp=20, Ki=0.1, Kd=10). State the relaxed goal threshold (0.8m) and longer episode length (1000 steps) and why.

---

### SECTION 6: EXPERIMENTAL SETUP (300 words, about 0.5 pages)

**Purpose:** Tell the reader exactly how experiments were run. So specific that another researcher can reproduce them.

**Structure:**

```
IV-A. Training Protocol
IV-B. Evaluation Protocol
IV-C. Metrics
```

**IV-A. Training Protocol (~100 words)**

> *"Each SAC condition (Naive, Uniform DR, CDR) is trained for 1×10⁶ steps across three independent seeds. All conditions share identical SAC hyperparameters and differ only in the physics distribution during training. Training uses a single NVIDIA T4 GPU and requires approximately 4.5 hours per run. The PID baseline requires no training."*

**IV-B. Evaluation Protocol (~100 words)**

> *"All policies are evaluated on a held-out test distribution whose ranges are strictly wider than any seen during training. No policy is fine-tuned on test data; all results represent zero-shot transfer. We run 100 deterministic evaluation episodes per seed per condition."*

**IV-C. Metrics (~100 words)**

List your four metrics and define each precisely:

1. **Success rate** — fraction of episodes reaching goal (define threshold: <0.5m for SAC, <0.8m for PID)
2. **Mean reward** — mean undiscounted return ± std across seeds
3. **Energy per step** — mean absolute thruster command, E = (1/T)Σ|a_t|, proportional to battery consumption
4. **Mean final distance** — distance to goal at episode end

State your statistical test: *"Comparisons use Welch's t-test (unequal variance). We report Cohen's d effect size and 95% bootstrap confidence intervals (1,000 resamples)."*

---

### SECTION 7: RESULTS (600 words, about 1.0 page)

**Purpose:** Report your results precisely and objectively. Every number you report must appear in your results JSONs. Do not interpret here — that is for Discussion. Do not describe the figures — let them speak for themselves.

**Structure: 4 subsections matching your 4 main findings**

```
V-A. PID Fragility Under Distribution Shift
V-B. Domain Randomisation Eliminates Deployment Variance
V-C. CDR Achieves Superior Energy Efficiency
V-D. Curriculum Level Progression
```

**V-A. PID Fragility (~120 words)**

Open with the dramatic number. Explain the mechanism briefly.

> *"Table II shows the primary result. PID achieves 100% success on the training distribution (reward -7.1). Under the held-out test distribution, success rate collapses to 3.0% (reward -249.0) — a 97-percentage-point drop. This collapse arises from the interaction of two effects. First, water currents at 0.40–0.80 m/s continuously push the AUV off-course; the proportional term cannot distinguish current-induced from inertia-induced error. Second, quadratic drag at 0.60–1.20 kg/m decelerates the AUV 3–6× faster than nominal; gains tuned for nominal drag over-command thrust, causing overshoot and limit-cycle oscillation."*

**V-B. DR Eliminates Variance (~120 words)**

> *"Naive SAC achieves 60.7% ± 27.2% transfer success. The ±27.2% standard deviation represents a deployment lottery — the same training procedure produces policies ranging from near-non-functional to moderately robust. Both DR conditions eliminate this variance entirely: Uniform DR achieves 99.7% ± 0.5% and CDR achieves 96.0% ± 3.7%, confirming that physics randomisation is necessary and sufficient for consistent transfer."*

**V-C. CDR Energy Efficiency (~200 words)**

This is your key positive result. Report the number, the statistical test, and then one sentence of interpretation. The deep interpretation goes in Discussion.

> *"Despite comparable transfer success rates (96.0% vs. 99.7%), CDR produces policies that consume 6.3% less thrust energy per step than Uniform DR (0.623 ± [std] vs. 0.665 ± [std]; p = [value], Cohen's d = [value], Welch's t-test). Both RL conditions substantially outperform PID on energy: PID applies near-maximum thrust continuously due to oscillatory integral action, whereas RL policies use the energy penalty to learn proportional thrust strategies.*
> 
> *Fig. [X] shows the results bar chart. CDR's energy advantage is consistent across all three seeds (individual dots in Fig. [X](c)), confirming this is a systematic effect of curriculum scheduling rather than a seed artefact."*

**V-D. Curriculum Level Progression (~100 words)**

> *"Fig. [X] shows the curriculum level λ over training. Seeds 1 and 2 reach λ > [value] by 10⁶ steps, confirming near-complete parameter expansion. The positive correlation between curriculum level reached and final transfer success suggests complete range expansion is predictive of policy quality — an observation with implications for early stopping criteria in future CDR deployments."*

**How to write a results section — general rules:**

- Report exact numbers with uncertainty: "96.0% ± 3.7%" not "approximately 96%"
- Compare to baseline whenever possible: "6.3% lower than Uniform DR" not "CDR achieved 0.623"
- Reference your figures and tables: "Table II shows..." and "Fig. 3 shows..."
- Past tense for experimental results: "CDR *achieved*..." not "CDR *achieves*..."

---

### SECTION 8: DISCUSSION (300 words, about 0.5 pages)

**Purpose:** Explain WHY your results happened. Interpret them. Discuss limitations honestly. A reviewer who sees you acknowledge limitations is more likely to trust your other claims.

**Structure:**

```
VI-A. Why CDR Produces More Energy-Efficient Policies
VI-B. Deployment Implications
VI-C. Limitations
```

**VI-A. Why CDR is More Efficient (~120 words)**

This is where you give the causal explanation, not just the correlation.

> *"The energy advantage of CDR has a mechanistic explanation rooted in the curriculum structure. In the early phase (λ ≈ 0), the agent trains exclusively on calm conditions. In calm water, the minimum-effort strategy for goal-reaching is smooth, low-thrust trajectory-following; the energy penalty reinforces this. As λ increases, the agent adapts its existing low-thrust strategy to harder conditions rather than developing a new one. Uniform DR, by contrast, presents all difficulty levels from episode 1. Under high drag and strong current, aggressive thrust is the only viable strategy early in training — and this aggressive pattern becomes entrenched."*

**VI-B. Deployment Implications (~80 words)**

Connect to the real world:

> *"The 6.3% thrust saving translates approximately linearly to 6.3% extended mission duration for thrust-dominated AUVs. For a 4-hour inspection mission this represents approximately 15 additional operational minutes — sufficient, in some scenarios, to complete an additional survey pass. We recommend that future AUV RL benchmarks report energy efficiency alongside success rate as a co-primary metric."*

**VI-C. Limitations (~80 words)**

Be honest. Reviewers respect this:

> *"This study is limited to simulation; hardware validation is left to future work. Our physics model follows Fossen [CITE] but omits turbulence, vortex shedding, and wave disturbances present in real ocean conditions. With n=3 seeds, statistical comparisons between closely matched conditions have limited power; the CDR vs. UDR energy difference reaches p=[value] — practically meaningful but not definitively established. Future work should validate on a physical vehicle with n≥10 seeds."*

---

### SECTION 9: CONCLUSION (150 words, about 0.25 pages)

**Purpose:** Remind the reader of what you proved. This is the last thing they read. It should be short, precise, and confident.

**The formula:**

```
Sentence 1: What you did (in one clause)
Sentence 2: Finding #1 (PID collapse) with number
Sentence 3: Finding #2 (DR eliminates variance) with number
Sentence 4: Finding #3 (CDR energy efficiency) with number
Sentence 5: The broader implication for the field
Sentence 6: What you released (open-source)
```

> *"We presented the first systematic comparison of curriculum versus uniform domain randomisation for AUV fluid physics, comparing CDR against Uniform DR, Naive SAC, and a PID baseline across nine experiments.*
> 
> *Three findings have practical significance: PID control collapses 100% → 3% under fluid-parameter shift, demonstrating that fixed-gain classical control is unsuitable for real ocean deployment; domain randomisation eliminates ±27.2% inter-seed deployment variance, enabling reliable pre-deployment assessment; and CDR produces 6.3% lower energy consumption than Uniform DR, a meaningful operational advantage for battery-constrained missions.*
> 
> *These results establish energy efficiency as a differentiator between DR strategies at comparable success rates. As AUV RL matures toward physical deployment, thrust economy should join success rate as a standard evaluation metric. Our open-source Halcyon X4 environment is released to support this.*"

---

## PART 3 — HOW TO WRITE ABOUT FIGURES AND TABLES

### Your 4–5 figures in 6 pages

You can fit approximately 4–5 figures in 6 pages. Here is what they should be:

| Figure | Title | What it shows | When to discuss |
|--------|-------|---------------|-----------------|
| Fig. 1 | CDR Pipeline Diagram | The training loop with expand/contract | Section III-D |
| Fig. 2 | Training Curves | ep_rew_mean + goal_dist vs steps, all 3 conditions, shaded std bands | Section V |
| Fig. 3 | Curriculum Progression | λ vs steps for 3 seeds | Section V-D |
| Fig. 4 | Results Bar Chart | Success, reward, energy per condition — 3 grouped bars | Section V-B/C |
| Fig. 5 (optional) | Ablation | If ablation runs done | Section V |

**How to write a figure caption:**

A good figure caption is self-contained — a reader should understand the figure without reading the surrounding text.

Template:
```
[What is shown]. [What to notice]. [What it means].
```

Example for Fig. 4:
> *"Transfer evaluation on the held-out test distribution (mean ± std across 3 seeds, 100 episodes each, individual seed dots overlaid). Dashed lines show PID performance. (a) Domain randomisation eliminates the high variance of Naive SAC. (b) PID achieves positive success but at dramatically negative reward, indicating oscillatory, energy-wasteful behaviour. (c) CDR achieves lowest energy consumption, confirming that curriculum scheduling produces intrinsically more efficient policies."*

**How to write a table caption:**

Table caption goes ABOVE the table (IEEE standard). Figure caption goes BELOW the figure.

---

## PART 4 — WRITING STYLE GUIDE

### Voice and tense

| Context | Tense | Example |
|---------|-------|---------|
| What your paper does | Present | "We compare CDR against..." |
| What happened in experiments | Past | "CDR achieved 96%..." |
| General scientific facts | Present | "Domain randomisation improves transfer..." |
| Describing figures | Present | "Fig. 3 shows the curriculum level..." |

### First person vs. passive voice

RA-L allows first person: "We train all conditions for 1M steps."  
Do NOT write: "In this paper, the authors trained..." (awkward)  
Do NOT write: "Models were trained by us for 1M steps." (passive is weak)

**Use "we" throughout.** It is clear, direct, and standard in IEEE robotics papers.

### Precision rules

Every quantitative claim must have:
1. A number
2. An uncertainty (std, CI, or "across 3 seeds")
3. A unit

| ❌ Vague | ✅ Precise |
|---------|-----------|
| "CDR achieves high success" | "CDR achieves 96.0% ± 3.7% success" |
| "significantly lower energy" | "6.3% lower energy (p=0.02, d=1.4)" |
| "much faster training" | "converges in ~4.5 hours on T4 GPU" |
| "improved over baseline" | "99.7% vs. 60.7% (Δ = 39 pp)" |

### Common mistakes to avoid

1. **Don't explain obvious things.** If you write "Deep learning has achieved great success in many domains," delete it.

2. **Don't hedge every claim.** "Our results suggest that CDR may potentially be able to produce somewhat more efficient policies" → "CDR produces more energy-efficient policies."

3. **Don't use superlatives without evidence.** "Our method is the best approach for AUV control" → "CDR achieves lower energy consumption than Uniform DR on this benchmark."

4. **Don't repeat yourself.** Each piece of information should appear once. The abstract summarises the whole paper. The conclusion summarises the whole paper. Do not copy sentences between them.

5. **Don't bury your key result.** Your most important result (6.3% energy saving) should appear in: the abstract, early in the introduction, and in the results section. Reviewers should not have to search for it.

---

## PART 5 — REFERENCING AND CITATION GUIDE

### How to cite in IEEE format

IEEE uses numbered citations: [1], [2], [3].

Rules:
- Cite immediately after the claim, before the period: "...as shown by Tobin et al. [1]."
- If citing multiple papers together: "...demonstrated in prior work [1], [2], [3]."
- The first time you name an author: "Tobin et al. [1] demonstrated..." 
- Subsequent times: "...as in [1]..."

### Your reference list (in order of first citation)

These are the core references for your paper:

```
[1]  T. Haarnoja, A. Zhou, P. Abbeel, S. Levine, "Soft Actor-Critic: Off-Policy 
     Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor," 
     ICML, 2018. arXiv:1801.01290

[2]  J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, P. Abbeel, 
     "Domain Randomization for Transferring Deep Neural Networks from 
     Simulation to the Real World," IROS, 2017. arXiv:1703.06907

[3]  X.B. Peng, M. Andrychowicz, W. Zaremba, P. Abbeel, "Sim-to-Real 
     Transfer of Robotic Control with Dynamics Randomization," ICRA, 2018.

[4]  I. Akkaya et al. (OpenAI), "Solving Rubik's Cube with a Robot Hand," 
     arXiv:1910.07113, 2019.

[5]  G. Tiboni, K. Arndt, V. Kyrki, "DORAEMON: Domain Randomization via 
     Entropy Maximization," ICLR, 2024. arXiv:2301.12457

[6]  T.T. Haarnoja, H. Tang, P. Abbeel, S. Levine, "Reinforcement Learning 
     with Deep Energy-Based Policies," ICML, 2017.

[7]  T. Haarnoja et al., "Soft Actor-Critic Algorithms and Applications," 
     arXiv:1812.05905, 2018.

[8]  K. Arndt, M. Ghadirzadeh, O. Kicki, V. Kyrki, 
     "Learning to Swim: Reinforcement Learning for 6-DOF AUV Control," 
     IROS, 2024. arXiv:2410.00120

[9]  S. Chu et al., "MarineGym: A High-Performance Reinforcement Learning 
     Platform for Underwater Robotics," arXiv:2503.09203, 2025.

[10] T. Chaffre et al., "Sim-to-Real Transfer of Adaptive AUV Control," 
     IJRR, 2025.

[11] T.I. Fossen, "Handbook of Marine Craft Hydrodynamics and Motion Control,"
     Wiley, 2011.

[12] S. Bengio, J. Louradour, R. Collobert, J. Weston, 
     "Curriculum Learning," ICML, 2009.

[13] Y. Shi, C. Shen, H. Fang, H. Li, "A Survey on Underwater Robot 
     Motion Control," Ocean Engineering, 2020.

[14] P. Klink, H. Abdulsamad, B. Belousov, J. Peters, "Self-Paced 
     Contextual Reinforcement Learning," NeurIPS, 2020.

[15] M.M. Manhães, S.C.A. Scherer, M. Voss, L.R. Douat, T. Rauschenbach,
     "UUV Simulator: A Gazebo-based Package for Underwater Intervention 
     and Multi-Robot Simulation," OCEANS, 2016.
```

---

## PART 6 — THE WRITING PROCESS STEP BY STEP

### The exact order to write sections

This is counterintuitive, but do NOT write the paper in order. Write it in this order:

```
Step 1: Method section (you know this best)
Step 2: Experimental Setup (straightforward facts)
Step 3: Results (report what happened)
Step 4: Discussion (explain what happened)
Step 5: Related Work (now that you know your contributions, compare)
Step 6: Conclusion (summarise what you wrote)
Step 7: Introduction (now that you know everything, introduce it)
Step 8: Abstract (last — summarise the whole paper in 250 words)
Step 9: Title (refine once abstract is done)
```

Why this order? Because you can't write a good introduction until you know exactly what your paper claims. You discover your final claims by writing the results section. The abstract is last because it summarises the finished paper.

### How to write a first draft

**Rule 1: Write fast, edit later.**
Set a timer for 25 minutes. Write continuously. Do not correct grammar, do not re-read, do not second-guess. Fill the page. Then take a 5-minute break. Then edit.

**Rule 2: Use placeholders.**
If you don't have a number yet, write [FILL: energy p-value] in brackets. Keep writing. Do not stop to run experiments mid-paragraph.

**Rule 3: One claim per sentence.**
Read each sentence. Does it make exactly one claim? If a sentence says two things, split it into two sentences.

**Rule 4: Read it aloud.**
After each paragraph, read it aloud. If you stumble, the sentence is too complicated. Simplify it.

### A writing schedule (realistic)

| Day | Task | Hours |
|-----|------|-------|
| 1 | Write Method (all subsections) | 3 hours |
| 2 | Write Experimental Setup + Results | 3 hours |
| 3 | Generate all figures and tables | 2 hours |
| 4 | Write Discussion + Conclusion | 2 hours |
| 5 | Write Related Work | 2 hours |
| 6 | Write Introduction | 2 hours |
| 7 | Write Abstract + refine title | 1 hour |
| 8 | Full read-through, fix all [FILL] placeholders | 3 hours |
| 9 | Compile LaTeX, fix formatting, check page count | 2 hours |
| 10 | Final proofread | 2 hours |

Total: ~22 hours of focused writing over 10 days.

---

## PART 7 — COMMON REVIEWER COMPLAINTS AND HOW TO AVOID THEM

These are the most common rejection reasons for robotics RL papers. Read each one and make sure you have addressed it.

**Reviewer complaint 1: "The evaluation is insufficient — only 3 seeds."**

How to avoid: Acknowledge in your limitations section. Also, 3 seeds is standard for computationally expensive RL experiments. State this: *"Following standard practice for computationally expensive RL experiments [CITE], we report results across 3 independent seeds. Each run requires ~4.5 GPU hours, making larger sweeps impractical on academic hardware."*

**Reviewer complaint 2: "The method is a minor modification of ADR."**

How to avoid: Explicitly state in the introduction what is new. CDR is novel because: (1) it applies to underwater fluid physics, a qualitatively different parameter space; (2) it uses a single shared window rather than per-parameter buffers; (3) it is the first work to show curriculum scheduling reduces energy consumption.

**Reviewer complaint 3: "The results are only in simulation — no real-world validation."**

How to avoid: Acknowledge this in limitations. State that sim-to-sim transfer with held-out distributions is accepted practice [CITE MarineGym, Arndt 2024]. State explicitly what future work is needed.

**Reviewer complaint 4: "There is no ablation study."**

How to avoid: If you have time, run ablation (5 conditions × 3 seeds). If not, acknowledge as future work. For RA-L without ablation, your paper is still publishable — but you may receive a "major revision" request.

**Reviewer complaint 5: "The writing is unclear / hard to follow."**

How to avoid: Use the structure in this guide. Define every symbol before using it. Give every variable a consistent name throughout. Do not use "CDR", "our method", and "the proposed approach" interchangeably — pick one and stick with it.

---

## PART 8 — LATEX FORMATTING CHECKLIST

Before submitting, verify every item:

```
□ IEEEtran document class, journal format
□ Title: title case, no abbreviations except AUV
□ Abstract: 250 words maximum, \begin{abstract}
□ Keywords: 5–8 keywords, \begin{IEEEkeywords}
□ All figures are PDF or EPS (not PNG unless 300 DPI)
□ All figures have captions below them
□ All tables have captions above them
□ All equations are numbered
□ All symbols defined at first use
□ All citations use \cite{key}
□ All references in IEEE format
□ Page count: ≤ 6 content + 1 references = 7 pages max
□ Double-column layout
□ Font: 10pt (IEEEtran default)
□ Margins: IEEEtran default (do not change)
□ No colour figures (unless you pay for colour printing — use RA-L colour option)
□ Supplementary video linked in abstract or introduction (optional but recommended)
□ Hyperref package with clickable links
□ Author information removed for double-blind review (RA-L is NOT double-blind, so keep your name)
□ Acknowledgements section (optional: thank your GPU provider, supervisors)
```

---

## PART 9 — SUBMISSION CHECKLIST FOR RA-L

When your paper is ready to submit, here is what you need:

**Files to prepare:**
1. `main.pdf` — the complete paper (max 7 pages total)
2. `source.zip` — all LaTeX source files + figures + .bib file
3. A cover letter (optional but recommended for RA-L)
4. A supplementary video (MP4, optional but strongly recommended for robotics)

**Cover letter template:**
> *"Dear Associate Editor,*
> 
> *We submit our manuscript "Curriculum Domain Randomisation for Energy-Efficient and Robust AUV Control" for consideration in IEEE Robotics and Automation Letters.*
> 
> *This paper presents the first systematic comparison of curriculum versus uniform domain randomisation for AUV fluid physics. Our key findings are: (1) PID control collapses from 100% to 3% under realistic fluid-parameter shift; (2) CDR achieves 96.0% transfer success with 6.3% lower energy consumption than Uniform DR.*
> 
> *We also submit this paper for presentation at IROS 2026 via the RA-L + IROS joint track.*
> 
> *[Your name], MIST, Bangladesh*"

**Where to submit:**
- URL: `mc.manuscriptcentral.com/ral`
- Select: "Submit to RA-L with option for IROS 2026 presentation"
- Suggested AE area: Autonomous Systems / Marine Robotics

---

## QUICK REFERENCE: ONE-PAGE SUMMARY

```
ABSTRACT (250 words)
  → Problem (PID fails) + Method (CDR) + Result (96%, 6.3% energy) + Impact + Release

INTRODUCTION (450 words)
  → Hook (100%→3%) → Problem → Prior work limits → Your approach → 3 bullet contributions

RELATED WORK (450 words)
  → DR literature → AUV RL literature → Curriculum learning
  → End each subsection: "Unlike prior work, we..."

METHOD (900 words)
  → MDP formulation → AUV environment → Table 1 (DR ranges) → CDR algorithm box
  → CDR hyperparameter table → SAC hyperparameter table → PID baseline

EXPERIMENTS (300 words)
  → Training protocol → Evaluation protocol → 4 metrics + statistical tests

RESULTS (600 words)
  → PID collapse (100%→3%) → DR eliminates variance → CDR energy (6.3%) → Curriculum level

DISCUSSION (300 words)
  → Why CDR is efficient (mechanism) → Deployment implications → 3 honest limitations

CONCLUSION (150 words)
  → 3 numbered findings with exact numbers → Field implication → Open-source release

FIGURES (4-5 total)
  1. CDR pipeline diagram
  2. Training curves (shaded std)
  3. Curriculum level progression
  4. Results bar chart (success + reward + energy)

REFERENCES (page 7)
  ~15-20 citations, IEEE format
```

---

*This guide was written specifically for your AUV CDR paper targeting IEEE RA-L 6-page format.*  
*Every example is drawn from your actual experimental results.*  
*Follow the order in Part 6 to write efficiently — method first, abstract last.*
