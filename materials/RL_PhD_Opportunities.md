# RL PhD Opportunities — Complete Reference Guide
**For:** EEE undergraduate at MIST, Bangladesh  
**Background:** RL, sim-to-real transfer, AUV robotics  
**Target:** PhD entry 2028, UK primary, global secondary  
**Goal:** High-salary industry job after PhD  

---

## How to use this guide

1. Read the RL domains section first — decide which areas genuinely interest you
2. Email professors in UK Tier 1 when your arXiv paper is live (mid 2026)
3. Apply formally in late 2027 for 2028 entry
4. Use the company list to target internships during PhD and jobs after

**One rule:** Always email professors directly before applying through portals. A portal application without prior contact is significantly weaker.

---

## RL Domains Worth Exploring

Your AUV work gives you a foundation in **sim-to-real transfer for physical systems**. That skill transfers across all of these domains. You are not starting from zero in any of them.

### 1. Robot Learning and Sim-to-Real Transfer
What you are already doing. Policies trained in simulation deployed on real robots. The gap between simulation and reality is the central unsolved problem. Directly applicable to arms, legged robots, drones, underwater vehicles, surgical robots.

**Why it matters:** Every robotics company deploying real robots needs this. Physical Intelligence, Figure, Boston Dynamics, Waymo all work on this daily.

**Key techniques to learn:** Domain randomisation, system identification, adaptive policies, meta-learning for fast adaptation.

---

### 2. Reinforcement Learning from Human Feedback (RLHF)
How modern LLMs (ChatGPT, Claude, Gemini) are aligned to be helpful and safe. RL is used to train a reward model from human preferences, then optimise the language model against that reward.

**Why it matters:** Every AI company building LLMs needs RLHF engineers. Highest salary ceiling in all of AI right now. Your RL fundamentals transfer directly — the algorithms are the same, the environment is different.

**Key techniques to learn:** Proximal Policy Optimisation (PPO), reward modelling, Constitutional AI, Direct Preference Optimisation (DPO).

---

### 3. Multi-Agent Reinforcement Learning (MARL)
Multiple agents learning simultaneously, cooperating or competing. Applications: robot swarms, autonomous vehicle fleets, game AI, economic modelling, drone coordination.

**Why it matters:** Real-world robotics increasingly involves multiple robots working together. AUV fleets for ocean survey is a direct extension of your current work.

**Key techniques to learn:** MADDPG, QMIX, MAPPO, communication protocols between agents.

---

### 4. Model-Based Reinforcement Learning (MBRL)
Learning a model of the environment (how actions affect states) and using it to plan. Much more sample efficient than model-free RL (like SAC). Critical for real-world applications where you cannot run millions of trial-and-error experiments on expensive hardware.

**Why it matters:** Your AUV training takes 4-5 hours on a T4 GPU. MBRL could potentially reduce that to minutes. Huge practical importance for physical robotics.

**Key techniques to learn:** World models, Dreamer, PETS, MuZero, Dyna-Q.

---

### 5. Offline Reinforcement Learning (Offline RL)
Learning from a fixed dataset of previously collected experience without interacting with the environment. Critical when online interaction is dangerous or expensive — medical robots, autonomous vehicles, industrial systems.

**Why it matters:** You cannot run a surgical robot for millions of trial-and-error steps. Offline RL learns from historical data. Growing fast in healthcare and industrial applications.

**Key techniques to learn:** Conservative Q-Learning (CQL), IQL, Decision Transformer, behaviour cloning.

---

### 6. Safe Reinforcement Learning
Ensuring RL agents satisfy safety constraints during both training and deployment. Critical for any real-world deployment — autonomous vehicles, medical devices, industrial robots, nuclear systems.

**Why it matters:** Regulators increasingly require safety guarantees for autonomous systems. The EU AI Act explicitly targets this. Big growth area for the next decade.

**Key techniques to learn:** Constrained MDPs, Lyapunov safety, CMDP, shielding, formal verification.

---

### 7. RL for Autonomous Vehicles
Self-driving cars, drones, underwater vehicles, autonomous ships. Your AUV work is already in this space. Extends naturally to aerial and surface vehicles.

**Why it matters:** Waymo, Cruise, Motional, Wayve — massive industry. RL increasingly central to motion planning and decision making. Your sim-to-real expertise is directly applicable.

---

### 8. RL for Healthcare and Biology
Drug discovery, protein folding, surgical robotics, treatment planning, prosthetics control. AlphaFold used RL components. Growing fast with massive funding.

**Why it matters:** Enormous social impact plus very strong funding. Academic and industry positions both well-funded. Less competition than robotics RL because the domain knowledge barrier is higher.

---

### 9. RL for Energy and Climate
Smart grid control, energy storage optimisation, building energy management, carbon capture process optimisation. Growing fast due to climate funding.

**Why it matters:** Governments worldwide are funding this heavily. Your EEE background gives you a genuine advantage — you understand power systems better than most ML researchers.

---

### 10. RL Theory
Convergence proofs, sample complexity, exploration theory, reward learning theory. More mathematical. Leads to faculty positions at top universities.

**Why it matters:** Theory people are rare and highly valued in academia. If you enjoy mathematics this is the path to a professorship. Not the highest industry salary but highest academic prestige.

---

## United Kingdom — Primary Target

### Tier 1 — Dream targets

---

#### Heriot-Watt University, Edinburgh

**Why:** Number one UK university for underwater and marine robotics. Ocean Systems Lab is world-renowned. Direct match to your AUV research.

**Website:** https://www.hw.ac.uk/research/ocean-systems-lab.htm

**Apply:** https://www.hw.ac.uk/study/postgraduate/research.htm

---

**Professor David Lane**  
*Director, Ocean Systems Lab. AUV autonomy, underwater robotics, marine AI.*  
**Email:** d.m.lane@hw.ac.uk  
**Profile:** https://www.hw.ac.uk/staff/dr-david-lane.htm  
**Why email him:** He has supervised AUV PhD students for 20+ years. Your paper on AUV sim-to-real transfer is a direct conversation opener. Reference his work on AUV mission planning.

---

**Professor Yvan Petillot**  
*Marine robotics, underwater perception, autonomous systems.*  
**Email:** y.petillot@hw.ac.uk  
**Profile:** https://www.hw.ac.uk/staff/prof-yvan-petillot.htm  
**Why email him:** Co-directs Ocean Systems Lab. Strong in perception and autonomy for underwater vehicles. Your simulation work complements his real-world deployment experience.

---

**Professor Sen Wang**  
*Robot learning, deep RL, sim-to-real transfer, autonomous driving.*  
**Email:** sen.wang@hw.ac.uk  
**Profile:** https://www.hw.ac.uk/staff/dr-sen-wang.htm  
**Why email him:** Newest relevant faculty. Works specifically on sim-to-real and deep RL. Your paper is directly in his research area. Most likely to respond positively to a strong preprint.

---

#### University of Edinburgh

**Why:** World-class informatics department. Strong RL and robot learning group. Close collaboration with Heriot-Watt.

**Website:** https://www.ed.ac.uk/informatics

**Apply:** https://www.ed.ac.uk/studying/postgraduate/degrees/research

---

**Professor Sethu Vijayakumar**  
*Robot learning, motor control, sim-to-real, adaptive systems.*  
**Email:** sethu.vijayakumar@ed.ac.uk  
**Profile:** https://homepages.inf.ed.ac.uk/svijayak/  
**Why email him:** One of the most cited robot learning researchers in Europe. Has sent students to DeepMind and OpenAI. Reference his work on DART (noise injection for sim-to-real).

---

**Professor Subramanian Ramamoorthy**  
*Autonomous systems, robot planning, uncertainty, transfer learning.*  
**Email:** s.ramamoorthy@ed.ac.uk  
**Profile:** https://homepages.inf.ed.ac.uk/sramamoo/  
**Why email him:** Works on generalisation and transfer for autonomous systems. Your CDR work connects to his interest in robust policy learning.

---

**Professor Michael Herrmann**  
*Computational neuroscience, RL, adaptive systems.*  
**Email:** michael.herrmann@ed.ac.uk  
**Profile:** https://www.inf.ed.ac.uk/people/staff/Michael_Herrmann.html  
**Why email him:** More theoretical RL focus. Good fit if you want to develop theory alongside experiments.

---

#### Imperial College London

**Why:** Top 10 globally for engineering. Dyson Robotics Lab is world-class. London location gives unparalleled industry access.

**Website:** https://www.imperial.ac.uk/electrical-engineering

**Apply:** https://www.imperial.ac.uk/study/pg/apply/

---

**Professor Andrew Davison**  
*Dyson Robotics Lab. Robot perception, SLAM, learning for robotics.*  
**Email:** a.davison@imperial.ac.uk  
**Profile:** https://www.doc.ic.ac.uk/~ajd/  
**Why email him:** Leads one of the best robotics labs in the world. More perception-focused but deeply interested in learning for physical systems.

---

**Professor Petar Kormushev**  
*Robot learning, RL for physical systems, human-robot interaction.*  
**Email:** p.kormushev@imperial.ac.uk  
**Profile:** https://www.imperial.ac.uk/people/p.kormushev.html  
**Why email him:** Direct match. Works on RL for physical robot control. Your sim-to-real work is exactly what his group does.

---

**Professor Bruno Lacerda**  
*Probabilistic planning, autonomous systems, uncertainty in robotics.*  
**Email:** b.lacerda@imperial.ac.uk  
**Profile:** https://www.imperial.ac.uk/people/b.lacerda.html  
**Why email him:** Strong in decision making under uncertainty for robots. Theoretical complement to your applied work.

---

### Tier 2 — Strong realistic targets

---

#### University of Bristol

**Website:** https://www.bristol.ac.uk/engineering/research/

**Apply:** https://www.bristol.ac.uk/study/postgraduate/research/

---

**Professor Arthur Richards**  
*Autonomous systems, control, UAV and UUV research, optimisation.*  
**Email:** arthur.richards@bristol.ac.uk  
**Profile:** https://www.bristol.ac.uk/people/person/Arthur-Richards-ded45ee0-fb01-4a93-bc84-53efe73ec39a/  
**Why email him:** Direct domain match. Works on autonomous underwater and aerial vehicles. Your AUV work maps directly onto his research.

---

**Professor Majid Mirmehdi**  
*Computer vision, robotics, machine learning.*  
**Email:** m.mirmehdi@bristol.ac.uk  
**Profile:** https://www.bristol.ac.uk/people/person/Majid-Mirmehdi  
**Why email him:** Strong group overall. Good for vision plus RL combination.

---

#### University of Southampton

**Why:** Strong in marine engineering. One of few UK universities with genuine AUV hardware and ocean research infrastructure.

**Website:** https://www.southampton.ac.uk/research/institutes-centres/maritime-engineering

**Apply:** https://www.southampton.ac.uk/study/postgraduate/research

---

**Professor Blair Thornton**  
*Underwater robotics, AUV systems, marine autonomy.*  
**Email:** b.thornton@soton.ac.uk  
**Profile:** https://www.southampton.ac.uk/people/5x2nqh/professor-blair-thornton  
**Why email him:** One of very few UK professors specifically working on AUV autonomy hardware. Your simulation work complements his real-system expertise. Strong collaboration potential.

---

**Professor Mahesan Niranjan**  
*Machine learning, probabilistic methods, neural networks.*  
**Email:** m.niranjan@soton.ac.uk  
**Profile:** https://www.ecs.soton.ac.uk/people/mn  
**Why email him:** Strong ML theory background. Good for more theoretical RL direction.

---

#### University of Manchester

**Website:** https://www.cs.manchester.ac.uk/research/

**Apply:** https://www.manchester.ac.uk/study/postgraduate-research/

---

**Professor Angelo Cangelosi**  
*Robot learning, developmental robotics, language grounding.*  
**Email:** angelo.cangelosi@manchester.ac.uk  
**Profile:** https://www.cs.manchester.ac.uk/~angelo/  
**Why email him:** Robot learning from a cognitive science angle. Good for MARL and human-robot interaction directions.

---

**Professor Timothy Sherrat**  
*Autonomous systems, swarm robotics.*  
**Email:** Check university directory  
**Profile:** https://www.manchester.ac.uk/research/  

---

#### University of Oxford

**Why:** Extremely competitive but worth one application. Strong in autonomous systems and safe RL.

**Website:** https://ori.ox.ac.uk

**Apply:** https://www.ox.ac.uk/admissions/graduate

---

**Professor Ingmar Posner**  
*Robot learning, autonomous driving, uncertainty.*  
**Email:** ingmar@robots.ox.ac.uk  
**Profile:** https://ori.ox.ac.uk/people/ingmar-posner/  
**Why email him:** Oxford Robotics Institute. Works on learning for autonomous systems. Very competitive to get into but worth trying.

---

**Professor Philip Torr**  
*Computer vision, deep learning, adversarial robustness.*  
**Email:** philip.torr@eng.ox.ac.uk  
**Profile:** https://torrvision.com  
**Why email him:** If you want to go toward vision plus RL combination.

---

#### University of Cambridge

**Why:** Extremely competitive. Worth one application if your paper is strong.

**Website:** https://www.eng.cam.ac.uk/research/academic-divisions/information-engineering

---

**Professor Joan Lasenby**  
*Robotics, signal processing, geometric algebra.*  
**Email:** jl@eng.cam.ac.uk  

**Professor Roberto Cipolla**  
*Computer vision, robotics.*  
**Email:** rc10001@cam.ac.uk  

---

#### University of Lincoln

**Why:** Surprisingly strong in field robotics. Underrated. Good funding availability.

**Website:** https://lcas.lincoln.ac.uk

---

**Professor Marc Hanheide**  
*Robot learning, human-robot interaction, agricultural robotics.*  
**Email:** mhanheide@lincoln.ac.uk  
**Profile:** https://staff.lincoln.ac.uk/mhanheide  
**Why email him:** LCAS lab is excellent. Field robotics — sim-to-real is central to their work. Very accessible and responsive to good applicants.

---

**Professor Grzegorz Cielniak**  
*Robot perception, field robotics, autonomous systems.*  
**Email:** gcielniak@lincoln.ac.uk  
**Profile:** https://staff.lincoln.ac.uk/gcielniak  

---

#### University of Sheffield

**Website:** https://www.sheffield.ac.uk/dcs/research

---

**Professor Noel Sharkey**  
*Robotics, autonomous systems, AI ethics.*  

**Professor Roderich Gross**  
*Swarm robotics, self-organising systems.*  
**Email:** r.gross@sheffield.ac.uk  
**Profile:** https://www.sheffield.ac.uk/dcs/people/academic/roderich-gross  

---

#### Cranfield University

**Why:** Postgraduate-only university. Strong defence and aerospace connections. Directly relevant to AUV applications in defence sector.

**Website:** https://www.cranfield.ac.uk/research

---

**Profile:** https://www.cranfield.ac.uk/centres/autonomous-systems  
Strong in autonomous systems overall. Check current faculty list for RL-adjacent supervisors.

---

### Tier 3 — Backup options

- **University of Leeds** — growing robotics group
- **King's College London** — ML and robotics growing fast
- **University College London (UCL)** — strong ML group, Professor Marc Deisenroth (robot learning) is excellent
- **University of Exeter** — growing ML, marine biology funding sometimes available
- **University of Glasgow** — robotics and autonomous systems group
- **Queen Mary University London** — good ML group, more accessible than UCL or Imperial

---

## Europe — Strong Alternatives to UK

### ETH Zurich, Switzerland

**Why:** Arguably the best robotics university in the world outside USA. Stipend CHF 4,000-5,000/month. Very competitive but worth applying.

**Website:** https://ethz.ch/en/research.html

**Apply:** https://www.ethz.ch/en/doctorate.html

---

**Professor Marco Hutter**  
*Robotic Systems Lab. Legged robotics, sim-to-real, RL for physical systems.*  
**Email:** mahutter@ethz.ch  
**Profile:** https://rsl.ethz.ch/the-lab/people/person-detail.MTIxOTEx.TGlzdC8yMDI0LC0xNTc1NTg3MDE=.html  
**Why email him:** His lab produced ANYmal — the most famous legged robot. sim-to-real is central to everything they do. Your curriculum DR work is directly relevant even though domain differs.

---

**Professor Davide Scaramuzza**  
*Robotics and Perception Group. Drone racing, visual RL, sim-to-real.*  
**Email:** sdavide@ifi.uzh.ch  
**Profile:** https://rpg.ifi.uzh.ch/people_scaramuzza.html  
**Note:** Based at University of Zurich, close ETH collaboration.  
**Why email him:** Pioneered sim-to-real for drone racing. Most sample-efficient sim-to-real work in aerial robotics. Direct methodological match.

---

**Professor Andreas Krause**  
*Learning and Adaptive Systems. Bayesian optimisation, safe RL, active learning.*  
**Email:** krausea@ethz.ch  
**Profile:** https://las.inf.ethz.ch/krausea  
**Why email him:** If you want to go toward safe RL or more theoretical direction.

---

### TU Delft, Netherlands

**Why:** Strong in robotics, affordable cost of living compared to UK, English language programmes, excellent industry connections.

**Website:** https://www.tudelft.nl/en/research

**Apply:** https://www.tudelft.nl/en/education/programmes/phd

---

**Professor Javier Alonso-Mora**  
*Autonomous multi-robot systems, motion planning, learning.*  
**Email:** j.alonso-mora@tudelft.nl  
**Profile:** https://www.autonomousrobots.nl  
**Why email him:** Multi-robot systems and motion planning with learning. MARL direction strongly fits here.

---

**Professor Robert Babuska**  
*RL, adaptive control, intelligent control systems.*  
**Email:** r.babuska@tudelft.nl  
**Profile:** https://www.tudelft.nl/staff/r.babuska  
**Why email him:** One of Europe's most experienced RL researchers. Strong theoretical background with practical applications.

---

**Professor Wendelin Bohmer**  
*Deep RL, policy gradient methods, function approximation.*  
**Email:** w.bohmer@tudelft.nl  
**Profile:** https://www.tudelft.nl/staff/w.bohmer  
**Why email him:** Deep RL theory and methods. Good for developing stronger theoretical foundations.

---

### University of Amsterdam, Netherlands

**Professor Max Welling**  
*Deep learning, probabilistic methods, geometric deep learning.*  
**Email:** m.welling@uva.nl  
**Profile:** https://staff.fnwi.uva.nl/m.welling/  
**Why email him:** World-class researcher. More theory-focused but excellent group to be part of.

---

### Technical University of Munich (TUM), Germany

**Why:** Free tuition, strong robotics, excellent industry connections to BMW, Siemens, MAN.

**Website:** https://www.tum.de/en/research

**Apply:** https://www.tum.de/en/studies/applying

---

**Professor Angela Schoellig**  
*Learning-based control, safe RL, autonomous drones.*  
**Email:** schoellig@tum.de  
**Profile:** https://www.tum.de/en/about-tum/people/professors/details/angela-schoellig  
**Why email her:** Safe RL and learning-based control for physical systems. Direct match to sim-to-real interests.

---

**Professor Matthias Niessner**  
*3D vision, neural rendering, robotics.*  
**Email:** niessner@tum.de  
**Profile:** https://niessnerlab.org  

---

### University of Freiburg, Germany

**Professor Wolfram Burgard**  
*Robot navigation, probabilistic robotics, deep learning for robots.*  
**Email:** burgard@informatik.uni-freiburg.de  
**Profile:** http://www2.informatik.uni-freiburg.de/~burgard/  
**Why email him:** One of the founding figures of modern mobile robotics. Navigation and learning for autonomous systems.

---

**Professor Abhinav Valada**  
*Robot learning, sim-to-real, autonomous systems.*  
**Email:** valada@informatik.uni-freiburg.de  
**Profile:** http://rl.uni-freiburg.de  
**Why email him:** Direct match — robot learning and sim-to-real transfer. Very active group.

---

### KTH Royal Institute of Technology, Sweden

**Why:** Strong robotics, Scandinavian quality of life, English language, gateway to Swedish tech industry (Ericsson, Volvo, Spotify).

**Website:** https://www.kth.se/en/forskning

---

**Professor Danica Kragic**  
*Robot vision and learning, manipulation, human-robot interaction.*  
**Email:** dani@kth.se  
**Profile:** https://www.kth.se/profile/dani  
**Why email her:** One of the most prominent robot learning researchers in Europe. Strong group, excellent mentorship reputation.

---

**Professor Patric Jensfelt**  
*Mobile robotics, SLAM, learning for navigation.*  
**Email:** patric@kth.se  
**Profile:** https://www.kth.se/profile/patric  

---

### INRIA / Sorbonne University, France

**Professor Olivier Sigaud**  
*Deep RL, continuous action spaces, intrinsic motivation.*  
**Email:** olivier.sigaud@sorbonne-universite.fr  
**Profile:** https://people.isir.upmc.fr/sigaud  
**Why email him:** RL theory and deep RL methods. Strong theoretical foundation.

---

## Asia-Pacific — Excellent Value, Growing Fast

### National University of Singapore (NUS)

**Why:** World top 15 university. Fully funded PhD. English language. Singapore is a major AI hub. Stipend SGD 2,500-3,500/month. Gateway to Southeast Asian tech industry.

**Website:** https://www.comp.nus.edu.sg/research/

**Apply:** https://www.comp.nus.edu.sg/programmes/pg/phd/

---

**Professor Harold Soh**  
*Robot learning, uncertainty, human-robot interaction, safe RL.*  
**Email:** harold@comp.nus.edu.sg  
**Profile:** https://haroldsoh.com  
**Why email him:** Directly relevant. Works on robot learning with uncertainty — critical for sim-to-real. Very active researcher, excellent mentor reputation.

---

**Professor Marcelo Ang**  
*Robotics, autonomous vehicles, manipulation.*  
**Email:** mpeangh@nus.edu.sg  
**Profile:** https://www.eng.nus.edu.sg/me/staff/ang-jr-marcelo-h/  

---

**Professor David Hsu**  
*Robotics, planning under uncertainty, autonomous vehicles.*  
**Email:** dyhsu@comp.nus.edu.sg  
**Profile:** https://www.comp.nus.edu.sg/~dyhsu/  
**Why email him:** POMDP planning and uncertainty — theoretical complement to your applied RL work.

---

### Nanyang Technological University (NTU), Singapore

**Professor Shuzhi Sam Ge**  
*Robotics, adaptive control, human-robot interaction.*  
**Email:** samge@nus.edu.sg  
**Profile:** https://www.ece.nus.edu.sg/stfpage/elesge/  

---

### KAIST, South Korea

**Why:** World-class engineering. Fully funded. Growing international presence. Strong industry connections to Samsung, Hyundai, LG.

**Website:** https://www.kaist.ac.kr/en/

**Apply:** https://admission.kaist.ac.kr/intl-graduate/

---

**Professor Donghyun Kim**  
*Legged robotics, RL for locomotion, sim-to-real.*  
**Email:** donghyun.kim@kaist.ac.kr  
**Profile:** https://www.dynamicrobot.kaist.ac.kr  
**Why email him:** Legged robotics and RL-based locomotion. Direct methodological match even though domain differs from AUV.

---

**Professor In So Kweon**  
*Computer vision, robotics, deep learning.*  
**Email:** iskweon@kaist.ac.kr  
**Profile:** https://rcv.kaist.ac.kr  

---

**Professor Jinwoo Shin**  
*Machine learning, deep learning theory.*  
**Email:** jinwoos@kaist.ac.kr  
**Profile:** https://alinlab.kaist.ac.kr/shin.html  

---

### University of Tokyo, Japan

**Professor Masatoshi Ishikawa**  
*High-speed robotics, vision, sensorimotor systems.*  

**Professor Jun Morimoto**  
*RL for motor control, humanoid robotics.*  
**Email:** morimoto@brain.riken.jp  
**Profile:** https://bicr.atr.jp/~morimoto/  

---

## North America — High Salary, Hard Visa

### University of Toronto, Canada

**Why:** Canada is more accessible visa-wise than USA. Strong AI ecosystem. Proximity to US tech companies.

**Professor Florian Shkurti**  
*Robot learning, sim-to-real, visual imitation.*  
**Email:** florian@cs.toronto.edu  
**Profile:** https://www.cs.toronto.edu/~florian/  
**Why email him:** Direct methodological match — sim-to-real transfer for physical robots. Strong publication record at top venues.

---

**Professor Animesh Garg**  
*Robot learning, manipulation, generalisation.*  
**Email:** animesh.garg@utoronto.ca  
**Profile:** https://animesh.garg.tech  

---

### McGill University, Canada

**Professor David Meger**  
*Robot learning, manipulation, sim-to-real.*  
**Email:** dmeger@cim.mcgill.ca  
**Profile:** https://www.cim.mcgill.ca/~dmeger/  

---

### University of British Columbia, Canada

**Professor Michiel van de Panne**  
*Physics-based animation, RL for locomotion.*  
**Email:** van@cs.ubc.ca  
**Profile:** https://www.cs.ubc.ca/~van/  

---

### Carnegie Mellon University, USA

**Why:** Best robotics university in the world. Very hard visa path from Bangladesh but worth knowing exists.

**Professor Deepak Pathak**  
*Self-supervised RL, curiosity-driven exploration, sim-to-real.*  
**Email:** dpathak@andrew.cmu.edu  
**Profile:** https://www.cs.cmu.edu/~dpathak/  

**Professor Katerina Fragkiadaki**  
*Robot learning, video prediction, physical reasoning.*  
**Email:** katef@cs.cmu.edu  
**Profile:** https://www.cs.cmu.edu/~katef/  

---

### UC Berkeley, USA

**Professor Sergey Levine**  
*Robot learning, offline RL, imitation learning.*  
**Email:** svlevine@eecs.berkeley.edu  
**Profile:** https://people.eecs.berkeley.edu/~svlevine/  
**Why email him:** The most influential robot learning researcher alive. Extremely competitive but your paper makes you a legitimate applicant.

---

## Australia — Underrated Option

### University of Sydney

**Professor Fabio Ramos**  
*Probabilistic robotics, sim-to-real, Bayesian deep learning.*  
**Email:** fabio.ramos@sydney.edu.au  
**Profile:** https://www.sydney.edu.au/engineering/about/our-people/academic-staff/fabio-ramos.html  
**Why email him:** Has published specifically on sim-to-real transfer using Bayesian methods. Direct methodological match.

---

**Professor Stefan Williams**  
*Marine robotics, AUV systems, underwater perception.*  
**Email:** stefan.williams@sydney.edu.au  
**Profile:** https://www.sydney.edu.au/engineering/about/our-people/academic-staff/stefan-williams.html  
**Why email him:** AUV domain expert. One of the leading underwater robotics researchers in the southern hemisphere. Your paper is a direct conversation starter.

---

### Queensland University of Technology (QUT)

**Professor Michael Milford**  
*Robot navigation, place recognition, RL.*  
**Email:** michael.milford@qut.edu.au  
**Profile:** https://staff.qut.edu.au/staff/michael.milford  

---

## Companies — Full Target List

### UK Companies — Priority

---

**Google DeepMind London**  
*The most important target. Largest concentration of RL talent outside academia.*  
**Website:** https://deepmind.google  
**What they do:** Fundamental RL research, robotics, protein folding, game AI, LLMs  
**Why you:** Your RL background plus physical systems experience matches their robotics team  
**How to get in:** PhD internship first (apply during year 2-3 of PhD), then full time  
**Salary:** £90k-£180k + equity  

---

**Wayve**  
*Autonomous driving with end-to-end learning. London headquartered.*  
**Website:** https://wayve.ai  
**What they do:** RL and imitation learning for self-driving. Sim-to-real central to their work.  
**Why you:** Sim-to-real transfer expertise directly applicable to their core challenge  
**Salary:** £70k-£130k  

---

**Oxa (formerly Oxbotica)**  
*Autonomous vehicle software. Oxford UK.*  
**Website:** https://oxa.tech  
**What they do:** Autonomous driving software platform. Heavy simulation use.  
**Why you:** Simulation and transfer expertise relevant  
**Salary:** £65k-£120k  

---

**Microsoft Research Cambridge**  
*Fundamental AI research. Cambridge UK.*  
**Website:** https://www.microsoft.com/en-us/research/lab/microsoft-research-cambridge/  
**What they do:** RL, robotics, healthcare AI, programming languages  
**Why you:** RL research background  
**Salary:** £80k-£150k  

---

**Dyson**  
*Yes, the vacuum company. Their robotics lab at Imperial is world-class.*  
**Website:** https://www.dyson.co.uk/careers  
**What they do:** Home robotics, autonomous cleaning robots, RL for manipulation  
**Why you:** RL for physical systems, sim-to-real expertise  
**Salary:** £55k-£100k  

---

**BAE Systems**  
*UK defence. Major AUV and autonomous systems programs.*  
**Website:** https://www.baesystems.com/en/careers  
**What they do:** Autonomous underwater vehicles, autonomous air systems, AI for defence  
**Why you:** AUV domain expertise is directly commercially valuable  
**Salary:** £45k-£85k + pension + stability  

---

**Rolls-Royce**  
*Autonomous ship and submarine research. Derby/London.*  
**Website:** https://careers.rolls-royce.com  
**What they do:** Autonomous marine vessels, predictive maintenance, RL for control  
**Why you:** Marine autonomy + RL combination  
**Salary:** £50k-£90k  

---

**Babcock International**  
*Naval systems. UK government contracts.*  
**Website:** https://www.babcockinternational.com/careers  
**What they do:** Naval engineering, submarine systems, autonomous underwater systems  
**Why you:** AUV and underwater systems expertise  

---

**Thales UK**  
*Defence technology. Autonomous systems.*  
**Website:** https://www.thalesgroup.com/en/united-kingdom/careers  
**What they do:** Naval systems, autonomous underwater vehicles, AI for defence  
**Why you:** Domain expertise match  

---

**DSTL (Defence Science and Technology Laboratory)**  
*UK government research. Directly funds AUV research.*  
**Website:** https://www.dstl.gov.uk/careers  
**What they do:** Defence research including underwater autonomy, AI, robotics  
**Why you:** AUV + RL + sim-to-real is exactly what they fund  
**Salary:** £40k-£75k + exceptional job security + pension  
**Note:** May require security clearance eventually — usually accessible to residents  

---

**Kongsberg Maritime UK**  
*Norwegian company, UK offices. World leader in AUV technology.*  
**Website:** https://www.kongsberg.com/maritime/careers/  
**What they do:** AUV systems, underwater robotics, ocean sensors  
**Why you:** Direct domain match. Your research is directly relevant to their products  

---

**Subsea 7**  
*Offshore energy, underwater robotics.*  
**Website:** https://www.subsea7.com/en/careers.html  
**What they do:** Underwater inspection, repair, installation using ROVs and AUVs  
**Why you:** AUV expertise, autonomy research  

---

**Fugro**  
*Geoscience and offshore. Heavy AUV user.*  
**Website:** https://www.fugro.com/careers  
**What they do:** Ocean survey, seabed mapping, infrastructure inspection using AUVs  
**Why you:** AUV domain knowledge  

---

**BP and Shell**  
*Offshore energy. Fund underwater autonomous systems research.*  
**BP Careers:** https://www.bp.com/en/global/corporate/careers.html  
**Shell Careers:** https://www.shell.co.uk/careers.html  
**What they do:** Offshore pipeline inspection, subsea asset management using AUVs  
**Why you:** AUV research directly reduces their inspection costs  
**Salary:** £55k-£100k  

---

### Global Companies — High Priority

---

**Physical Intelligence (pi)**  
*The most exciting robotics company in the world right now.*  
**Website:** https://www.physicalintelligence.company  
**Location:** San Francisco, USA  
**What they do:** General-purpose robot learning. π0 policy trained on diverse robot data.  
**Why you:** RL for physical systems, generalisation, sim-to-real  
**Salary:** $200k-$350k total comp  

---

**Figure AI**  
*Humanoid robots. BMW partnership.*  
**Website:** https://www.figure.ai/careers  
**Location:** Sunnyvale, USA  
**What they do:** Humanoid robot learning, RL-based control, warehouse automation  
**Why you:** RL for physical systems  
**Salary:** $180k-$300k total comp  

---

**Boston Dynamics**  
*Real robot deployment. Spot and Atlas.*  
**Website:** https://www.bostondynamics.com/careers  
**Location:** Waltham USA, offices expanding  
**What they do:** Legged robot control, RL for locomotion, autonomous inspection  
**Why you:** Sim-to-real transfer expertise directly applicable  
**Salary:** $150k-$250k  

---

**Agility Robotics**  
*Humanoid robots for warehouses. Amazon partnership.*  
**Website:** https://www.agilityrobotics.com/careers  
**Location:** Oregon, USA  
**What they do:** Bipedal robot locomotion, RL-based control, warehouse deployment  

---

**Saildrone**  
*Autonomous ocean vehicles. Direct AUV match.*  
**Website:** https://www.saildrone.com/careers  
**Location:** Alameda, USA  
**What they do:** Autonomous surface and underwater vehicles for ocean data collection  
**Why you:** Your AUV research is directly relevant to their core product  

---

**Oceaneering International**  
*Subsea robotics. Global presence.*  
**Website:** https://www.oceaneering.com/careers/  
**Location:** Houston USA, Aberdeen UK, multiple global offices  
**What they do:** ROV and AUV operations, subsea intervention, autonomous inspection  
**Why you:** AUV domain expertise  

---

**Teledyne Marine**  
*AUV manufacturer.*  
**Website:** https://www.teledynemarine.com/careers  
**Location:** Multiple global offices  
**What they do:** AUV design, manufacture, and deployment  
**Why you:** Your research directly applies to their products  

---

**Waymo**  
*Self-driving cars. Alphabet subsidiary.*  
**Website:** https://waymo.com/careers/  
**Location:** Mountain View, USA  
**What they do:** Autonomous driving, heavy simulation use, sim-to-real  
**Why you:** Sim-to-real expertise transfers across domains  
**Salary:** $200k-$400k total comp  

---

**Amazon Robotics**  
*Warehouse robotics. Massive scale.*  
**Website:** https://www.amazon.jobs/en/teams/amazon-robotics  
**Location:** Multiple, UK offices  
**What they do:** Autonomous warehouse robots, RL for manipulation and navigation  
**Why you:** RL for physical systems  
**Salary:** £80k-£150k UK, $180k-$300k USA  

---

**ECA Group**  
*French AUV manufacturer. European presence.*  
**Website:** https://www.ecagroup.com/en/careers  
**What they do:** AUV and drone systems for defence and industry  
**Why you:** AUV domain match  

---

## Summer Research Programs — Apply for 2027

These are funded international research experiences during your undergraduate degree. Each one strengthens your CV dramatically.

| Program | Country | Deadline | Link | Your fit |
|---------|---------|----------|------|----------|
| DAAD WISE | Germany | Nov-Jan | https://www.daad.de/en/studying-in-germany/scholarships/daad-scholarships/wise/ | Very high — engineering undergrad |
| NUS Research Attachment | Singapore | Varies | https://www.nus.edu.sg/registrar/programmes/non-graduating | Very high |
| KAIST Undergraduate Research | South Korea | Feb-Mar | https://admission.kaist.ac.kr | High |
| Mitacs Globalink | Canada | Sep-Oct | https://www.mitacs.ca/globalink | High |
| ETH Zurich Summer Research | Switzerland | Dec-Jan | https://ethz.ch/en/studies/non-degree-courses/summer-research-fellowship.html | High |
| OIST Research Internship | Japan | Nov | https://www.oist.jp/research-internship | High |
| Charpak Research Internship | France | Jan-Feb | https://www.inde.campusfrance.org/charpak-lab-scholarship | Medium |

Apply to at least 4 of these for summer 2027. Your arXiv paper makes you competitive for all of them.

---

## Priority Email Schedule — Mid 2026

When your paper is on arXiv, email in this order over 3 weeks.

**Week 1 — UK Tier 1**
1. Professor Sen Wang — Heriot-Watt (most likely to respond)
2. Professor David Lane — Heriot-Watt (most domain-relevant)
3. Professor Yvan Petillot — Heriot-Watt

**Week 2 — UK Tier 1 and international**
4. Professor Sethu Vijayakumar — Edinburgh
5. Professor Florian Shkurti — Toronto (sim-to-real match)
6. Professor Harold Soh — NUS

**Week 3 — UK Tier 2 and stretch**
7. Professor Blair Thornton — Southampton
8. Professor Arthur Richards — Bristol
9. Professor Abhinav Valada — Freiburg (sim-to-real match)
10. Professor Stefan Williams — Sydney (AUV match)

---

## Email Template

Subject: `Prospective PhD (2028) — RL sim-to-real transfer for autonomous systems`

```
Dear Professor [Name],

I am a third-year EEE undergraduate at MIST, Bangladesh, 
planning to apply for PhD positions in 2028.

I have been conducting independent research on curriculum 
domain randomisation for AUV sim-to-real transfer and 
recently posted a preprint: [arXiv link]

Your work on [specific paper title] directly connects to 
my research on [specific connection]. I am particularly 
interested in extending this toward [one sentence on 
future direction relevant to their work].

I would be grateful to know whether you anticipate 
funded PhD positions opening in 2028, and whether my 
background might be a fit for your group.

GitHub: [link]

Thank you for your time.
[Your name]
```

Keep it short. Personalise the middle paragraph for every professor. Never send the same email twice.

---

## Key Facts to Remember

**Funding:** Never accept an unfunded PhD offer. Always confirm tuition plus stipend before accepting.

**Stipends by country (approximate monthly):**
- UK: £1,600-£1,800
- Germany: €1,400-€2,000
- Netherlands: €2,000-€2,500
- Switzerland: CHF 4,000-5,000
- Singapore: SGD 2,500-3,500
- South Korea: KRW 1,500,000-2,500,000

**Duration:**
- UK, Europe, Singapore: 3-4 years
- USA: 5-6 years (avoid for speed)
- South Korea: 4-5 years

**UK Graduate Visa:** After finishing UK PhD you get 3 years automatic right to work in UK. No job offer needed before graduation. Very valuable.

**Application timeline:**
- Late 2026: Email professors, build relationships
- Oct-Dec 2027: Submit formal applications
- Jan-Apr 2028: Interviews
- Sep 2028: Start PhD

---

*Last updated: March 2026*  
*Next update: When seed=0 results are confirmed after bug fix*
