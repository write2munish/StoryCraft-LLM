# 🤝 Module 08: Post-Training Phase 2 - Alignment via DPO (Direct Preference Optimization)

> **Central Anchor:** Aligning our instruction-following story model with human values (kindness, sharing, child safety) using closed-form Direct Preference Optimization (DPO) and saving `checkpoints/storycraft_aligned.pt`.

---

## 🎯 What to Expect
- **What You Will Build & Run:** You will execute `demo_alignment_dpo.py`, loading `checkpoints/storycraft_instruct.pt`, curating preference pairs (chosen wholesome story $y_w$ vs rejected rude ending $y_l$), computing closed-form DPO loss, and updating the policy model.
- **What You Will See:** The terminal displays preference comparisons, log-ratio probabilities between the policy $\pi_\theta$ and frozen reference model $\pi_{\text{ref}}$, loss convergence, and saves `checkpoints/storycraft_aligned.pt`.
- **What You Will Observe:** You will observe how DPO analytically steers model probabilities toward kindness and sharing without requiring reinforcement learning, critics, or separate reward models.
- **Run Command:**
  ```bash
  python3 modules/08_alignment_dpo/demo_alignment_dpo.py
  ```

---

## 💡 The Concept Explained Simply
An instruction-tuned model follows orders, but it doesn't know what makes a story *good*, *safe*, or *wholesome* for a young child. If a prompt says: *"Lily took Barnaby's toy"*, the model might generate: *"Barnaby bit Lily and she cried alone."* Technically that is a continuation, but it is terrible for a children's storybook!

**How Do We Align LLMs with Human Values?**
- Historically, companies used **RLHF (Reinforcement Learning from Human Feedback)**. RLHF required training 4 separate models at once (Actor, Critic, Reward Model, Reference Model) and using unstable reinforcement learning algorithms (PPO) that frequently collapsed.
- **Direct Preference Optimization (DPO)** is a groundbreaking mathematical discovery: we show the model pairs of stories:
  - **Chosen ($y_w$):** Lily shared her cookies, and everyone smiled warmly.
  - **Rejected ($y_l$):** Lily was selfish and yelled at her friends.
- DPO has a single, closed-form loss equation that increases the probability of $y_w$ while decreasing the probability of $y_l$, using the frozen instruction model as a reference anchor. It gives all the benefits of RLHF with the stability and simplicity of standard cross-entropy!

---

## 🔬 Core Theory: Direct Preference Optimization (DPO)
DPO mathematically demonstrates that the optimal policy under the Bradley-Terry preference model can be derived directly without an explicit reward function:
$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

### Key Mechanics:
- **$\pi_\theta$ (Trainable Policy Model):** The model being updated to prefer wholesome endings.
- **$\pi_{\text{ref}}$ (Frozen Reference Model):** An identical copy of `storycraft_instruct.pt` whose weights never change.
- **$\beta$ (Implicit Temperature):** Controls how strictly the policy is constrained to stay close to the reference model (typically $0.1$). It prevents the model from generating repetitive gibberish just to "game" the preference objective.

---

## 💻 File Structure & Runnable Components
```
modules/08_alignment_dpo/
├── README.md                 # This module guide
├── demo_alignment_dpo.py     # Runnable DPO alignment & policy optimization engine
└── tests.py                  # 2-test unit suite verifying DPO loss formulation & margin behavior
```

---

## 🔑 Key Pedagogical Takeaways
1. **RL-Free Alignment:** DPO achieves the exact same mathematical objective as RLHF/PPO while eliminating reward model training and policy instability.
2. **Implicit Reward Function:** The ratio $\log(\pi_\theta / \pi_{\text{ref}})$ acts as the implicit reward function.
3. **Reference Anchoring Prevents Drift:** Without $\pi_{\text{ref}}$ penalizing policy drift via KL divergence, language models quickly degenerate into degenerate token repetitions.
4. **Bridge to Module 09:** With Base, Instruct, and Aligned checkpoints saved, Module 09 benchmarks all three models across fluency, diversity, and safety rubrics.

---

## 🧪 Running the Code & Unit Tests
```bash
# Run the interactive DPO alignment demo
python3 modules/08_alignment_dpo/demo_alignment_dpo.py

# Run the automated unit tests
python3 modules/08_alignment_dpo/tests.py
```
