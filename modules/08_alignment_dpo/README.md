# 🧭 Module 08: Post-Training Phase 2 - Alignment via DPO

> **Central Anchor in this Module:** Aligning `StoryCraft-Instruct` using Direct Preference Optimization (DPO) so it prefers kind, uplifting stories with moral resolutions over boring, scary, or abrupt endings.

---

## 🎯 Learning Objectives
1. Understand the alignment problem: why supervised fine-tuning cannot penalize negative or unsafe behaviors effectively.
2. Demystify the transition from complex RLHF (PPO) to closed-form Direct Preference Optimization (DPO).
3. Curate pairwise preference datasets: `(prompt, chosen_story, rejected_story)`.
4. Train `storycraft_aligned.pt` using the DPO loss function.

---

## 🔬 Core Theory: Direct Preference Optimization (DPO)

In traditional RLHF (Reinforcement Learning from Human Feedback), you must:
1. Train a separate Reward Model on pairs of answers.
2. Run PPO with an Actor, Critic, and Reference policy (prone to training instability and memory exhaustion).

**DPO (Rafailov et al., 2023)** bypassed the reward model entirely by proving that an optimal policy implicitly defines its own reward:
$$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

### The DPO Loss Function:
$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

- $\pi_\theta$: The model we are training.
- $\pi_{\text{ref}}$: The frozen SFT model from Module 06 (prevents the model from drifting into degenerate output).
- $y_w$: The preferred (winning) story (wholesome, well-paced, constructive).
- $y_l$: The rejected (losing) story (abrupt ending, repetitive, or unkind).
- $\beta$: Temperature parameter (typically $0.1 - 0.5$).

---

## 💻 Lab Exercises
- Implement `train_dpo.py`:
  - Load frozen reference model $\pi_{\text{ref}}$ and trainable policy $\pi_\theta$.
  - Compute log-probabilities for $(x, y_w)$ and $(x, y_l)$ under both models.
  - Implement the DPO loss equation and optimize $\theta$.
  - Test the resulting aligned model: verify that when prompted with challenging scenarios, the model steers towards positive, safe resolutions.
