## II. Related Work

### A. Domain Randomisation for Sim-to-Real Transfer

Tobin et al. [CITE] first demonstrated domain randomisation for visual
sim-to-real transfer in robotic grasping. Akkaya et al. [CITE] extended
this to dexterous manipulation via Automatic Domain Randomisation (ADR),
which expands parameter ranges based on policy performance — the closest
prior work to our approach. Tiboni et al. [CITE] propose DORAEMON, which
maximises entropy over the randomisation distribution. Unlike these works,
which target manipulation and locomotion tasks, we apply curriculum DR
specifically to underwater fluid dynamics — a parameter space with
distinct non-linear structure (quadratic drag) that we show benefits
from curriculum ordering.

### B. RL for Autonomous Underwater Vehicles

[FILL after reading MarineGym paper]
Key point to make: prior AUV RL work uses fixed DR ranges or no DR at all.

### C. Multi-Fidelity Sim-to-Real Evaluation

[FILL after reading UUV Simulator paper]
Key point: justify why MuJoCo → UUV Simulator is a valid evaluation protocol.