# 💬 Module 07: Post-Training Phase 1 - Instruction Tuning (SFT) & LoRA from Scratch

> **Central Anchor:** Converting our raw text-completing base model into a polite, prompt-following story assistant (`StoryCraft-Instruct`) using ChatML formatting and Parameter-Efficient Fine-Tuning via LoRA.

---

## 🎯 What to Expect
- **What You Will Build & Run:** You will execute `demo_sft_lora.py`, loading `checkpoints/storycraft_base.pt`, formatting dialogue pairs into ChatML markup, applying target-only loss masking, and training low-rank adapter matrices ($A$ and $B$).
- **What You Will See:** The terminal displays ChatML prompt formatting, counts trainable vs frozen parameters (proving $<1\%$ of weights update), executes a LoRA fine-tuning step, and saves `checkpoints/storycraft_instruct.pt`.
- **What You Will Observe:** You will observe why Base models fail to answer questions without SFT, and how LoRA fine-tuning adapts attention projections with negligible memory overhead.
- **Run Command:**
  ```bash
  python3 modules/07_instruction_tuning_sft/demo_sft_lora.py
  ```

---

## 💡 The Concept Explained Simply
A raw base model is simply an auto-completer. If you type:
> *"Write a bedtime story about a kitten named Whiskers."*

The base model doesn't know you are asking for help! It might treat your prompt as a school exam and reply:
> *"2. Write a story about a puppy named Barky. 3. Write a story about a duck."*

**Instruction Tuning (SFT) is how we teach the model conversational etiquette:**
1. **ChatML Formatting:** We wrap dialogues in clear role tags (`<|im_start|>user`, `<|im_start|>assistant`). This teaches the model: *"When the user speaks, my job is to answer as the assistant!"*
2. **Target-Only Loss Masking:** We only calculate loss and backpropagation on the *assistant's response tokens*. We never penalize the model for the words in the user's prompt!
3. **LoRA (Low-Rank Adaptation):** Modifying all 820,000 weights of our model is slow and risks destroying its language abilities. Instead, LoRA freezes the original model completely and injects two tiny adapter matrices ($A$ and $B$) next to each attention layer. It achieves full fine-tuning performance while training less than $1\%$ of the parameters!

---

## 🔬 Core Theory: Low-Rank Adaptation (LoRA)

### Mathematical Formulation:
For a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA decomposes the weight update $\Delta W$ into the product of two low-rank matrices:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
Where:
- $A \in \mathbb{R}^{r \times k}$ is initialized with Gaussian noise $\mathcal{N}(0, \sigma^2)$.
- $B \in \mathbb{R}^{d \times r}$ is initialized to zero, ensuring $\Delta W = 0$ at the start of training.
- $r \ll \min(d, k)$ is the rank (e.g., $r = 4$).
- $\alpha$ is a constant scaling hyperparameter (e.g., $\alpha = 16$).

### Target-Only Loss Masking:
$$\mathcal{L}_{\text{SFT}} = -\sum_{t \in \text{Assistant Tokens}} \log P(y_t \mid x_{<t})$$
All prompt tokens are assigned `target = -100` (`ignore_index` in PyTorch `CrossEntropyLoss`).

---

## 💻 File Structure & Runnable Components
```
modules/07_instruction_tuning_sft/
├── README.md                 # This module guide
├── demo_sft_lora.py          # Runnable SFT ChatML pipeline & LoRA adapter engine
└── tests.py                  # 3-test unit suite verifying LoRA forward pass & weight merging
```

---

## 🔑 Key Pedagogical Takeaways
1. **The Base-to-Assistant Transition:** SFT transforms raw text continuation into reliable prompt obedience.
2. **Loss Masking is Mandatory:** If you do not mask user prompt tokens, the model wastes capacity memorizing how users ask questions instead of learning how to answer.
3. **LoRA Parameter Efficiency:** Decomposing weight updates into low-rank matrices ($B \cdot A$) dramatically slashes VRAM and optimizer state memory.
4. **Zero Latency at Inference:** At deployment, LoRA weights can be mathematically added directly into the base weights ($W_0 + \Delta W$), incurring zero inference overhead!
5. **Bridge to Module 08:** SFT teaches the model to follow instructions; Module 08 aligns it with human values, safety, and kindness using Direct Preference Optimization (DPO).

---

## 🧪 Running the Code & Unit Tests
```bash
# Run the interactive SFT & LoRA demo
python3 modules/07_instruction_tuning_sft/demo_sft_lora.py

# Run the automated unit tests
python3 modules/07_instruction_tuning_sft/tests.py
```
