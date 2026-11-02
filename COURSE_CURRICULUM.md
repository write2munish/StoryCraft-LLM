# 📚 StoryCraft-LLM: Complete Course Curriculum

> **Building a Modern Large Language Model from First Principles**

> **A First-Principles Journey Through LLM Architecture, Pre-training, Fine-Tuning, Alignment, and Deployment**  
> *Central Unifying Anchor: The Tiny Story Generator ("StoryCraft-LLM")*  
> *Enriched with battle-tested patterns, modern architectural evolutions, and engineering insights from Sebastian Raschka's `LLMs-from-scratch`.*

---

## 🌟 Executive Overview & Pedagogical Philosophy

Most LLM courses suffer from one of two extremes:
1. **Too High-Level:** Calling `AutoModelForCausalLM.from_pretrained(...)` and tweaking HuggingFace black boxes without understanding tokenization, attention mathematics, KV caching, or loss dynamics.
2. **Too Abstract / Overwhelming:** Diving into 70B parameter distributed training, Megatron-LM tensor parallelism, and multi-node clusters where students cannot train or debug models on their personal machines.

### The "TinyStories" Breakthrough Anchor
This course solves both problems by anchoring every single module around one cohesive, tangible product: **StoryCraft-LLM (The Tiny Story Generator)**.

Inspired by the landmark Microsoft Research paper *"TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"* (Eldan & Li, 2023) and enhanced by the rigorous, bottom-up implementations of Sebastian Raschka's *Build a Large Language Model (From Scratch)*:
- Small models with **5M–15M parameters** can master fluent grammar, vocabulary, character consistency, and storytelling if trained on a focused domain.
- Students can train, evaluate, and fine-tune their model locally on an **Apple Silicon Mac (M1/M2/M3/M4)**, a single **NVIDIA RTX GPU**, or a free **Google Colab** tier in **under 45 minutes**.
- Every single line of code—from BPE tokenization to Multi-Head Attention, RoPE, SwiGLU, AdamW, LoRA adapters, SFT, DPO, and KV-caching—is written in pure PyTorch and Python.

---

## 🗺️ The StoryCraft Evolution Matrix

Across the modules, students watch the **exact same model** evolve through distinct developmental stages:

| Stage / Module | What We Build | Model State | Sample Generated Output / Capability |
| :--- | :--- | :--- | :--- |
| **00. Foundations** | N-Gram & Markov Baseline | Statistical Count Table | `"Once upon a time dog ball the the saw saw."` |
| **01. Tokenization** | Custom Byte-Pair Encoder | Token dictionary & ID maps | `[104, 892, 334, 12]` $\to$ Tokens & Special tags |
| **02. Embeddings** | Vector lookup & RoPE | Continuous geometry | Embeddings capture word analogies: `king - man + woman = queen` |
| **03. Architecture** | Modern Transformer Blocks | Untrained random weights | `"x7# qw 9! lkjsd &&% mmz0"` (pure static entropy) |
| **04. Pre-training** | Autoregressive Engine | Minimizing cross-entropy loss | `"Once upon a time, there was a little boy named Tim. He had a red ball."` |
| **05. Sampling & Weights** | Temperature, Top-p, Weight Loading | Controlled generation + State Dict mapping | Coherent paragraphs; loading open weights into scratch model |
| **06. Classification** | Story Classifier Head ($d \to K$) | Discriminative LLM Backbone | Input: Story $\to$ Output: `Genre: Adventure`, `Age: 3-5`, `Safe: True` |
| **07. Instruction & LoRA** | SFT with ChatML + LoRA from Scratch | Prompt-conditioned + Modular Adapters | **Prompt:** `Write a story about a frog who wants to fly.`<br>**Response:** `"Once upon a lily pad..."` (Trained with only ~50k LoRA params!) |
| **08. Alignment (DPO)** | Direct Preference Optimization | Value-aligned model | Prefers constructive, kind, and safe resolutions over abrupt, sad, or nonsensical endings |
| **09. Evaluation** | Automated Evaluation & Benchmarks | Verified model metrics | Perplexity $< 4.2$, Grammar: $9.4/10$, Coherence: $8.9/10$, LLM-as-a-Judge |
| **10. KV Cache & Speed** | KV Cache & Quantization | Low-latency inference ($> 4\times$ speedup) | Real-time token streaming ($> 120\text{ tok/s}$) on consumer CPU |
| **11. Capstone App** | Full-stack Interactive Storybook | Production artifact | Interactive Web UI with prompt controls, story branching, and moral tuning |

---

## 📖 Module-by-Module Detailed Curriculum

```
===================================================================================
 PREREQUISITE: Front-Loaded PyTorch Primer (Tensor Mechanics & Autograd)
===================================================================================
```
### 🎯 Learning Objectives
- Master tensor creation, device placement (`cpu`, `cuda`, `mps`), memory layouts, and non-contiguous stride mechanics.
- Understand zero-copy views (`.view()`, `.reshape()`) and broadcasting rules essential for multi-head attention and RoPE tensors.
- Demystify Autograd dynamic computation graphs (DAG), reverse-mode automatic differentiation, and `torch.no_grad()`.
- Construct custom `nn.Module` classes with explicit parameter tracking, registration of persistent buffers (`register_buffer`), and state serialization.

### 🧠 Core Concepts & Theory
1. **The PyTorch Mental Model:**
   - Tensors as continuous C/CUDA memory buffers overlaid with multidimensional strides: $\text{index} = \sum_i \text{coord}[i] \times \text{stride}[i]$.
   - View operations vs memory re-allocation (`.contiguous()`).
2. **Broadcasting in Modern Attention & RoPE:**
   - Aligning dimensions: `[1, 1, T, D]` with `[B, H, T, D]` without duplicating memory.
3. **Autograd DAG & Memory Safety:**
   - Dynamic graph construction during the forward pass; topological reverse-mode traversal during `.backward()`.
   - Gradient accumulation vs `.zero_grad(set_to_none=True)`.
   - Disabling graph tracking during evaluation and inference using `torch.no_grad()` or `torch.inference_mode()`.
4. **Parameters vs Buffers:**
   - Learnable weights (`nn.Parameter`) vs non-trainable state like RoPE frequencies or causal masks (`register_buffer`).

### 💻 Hands-on Lab & Verification
- **Directory:** `pytorch_primer/` (also symlinked as `modules/primer_pytorch`)
- **Code:** `pytorch_primer/demo_pytorch_primer.py`
- **Verification:** `pytorch_primer/tests.py` (5 unit tests verifying strides, broadcasting, autograd gradients, no-grad contexts, and registered buffers).

---

```
===================================================================================
 MODULE 00: Orientation & The Mental Model of Language Modeling
===================================================================================
```
### 🎯 Learning Objectives
- Formulate text generation as probabilistic autoregressive sequence modeling.
- Understand the joint distribution decomposition via the Chain Rule of Probability.
- Implement an N-gram baseline to discover the "Curse of Dimensionality" and why neural networks are necessary.

### 🧠 Core Concepts & Theory
1. **What is an LLM really doing?** Next-token probability distribution:
   $$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, w_2, \dots, w_{t-1})$$
2. **The Discrete Bottleneck:** Why language is fundamentally harder to optimize than continuous signals (images/audio).
3. **Statistical vs. Distributed Representations:**
   - Bigram & Trigram models: counting transitions $C(w_t \mid w_{t-1}) / C(w_{t-1})$.
   - Sparsity: What happens when an unseen combination appears?
   - Exponential explosion: Storing a 5-gram model for a $5,000$ vocabulary requires $5,000^5 \approx 3.1 \times 10^{18}$ entries!

### 💻 Hands-on Lab: The Bigram Storyteller
- **File:** `modules/00_foundations/bigram_story_generator.py`
- Build a pure Python bigram and trigram model on a toy story corpus (1,000 children's stories).
- Generate stories from prompts like `"Once upon a time"`.
- Analyze where it fails: repetition loops, lack of long-term memory, inability to retain character identity beyond 2 words.
- **Verification:** Unit tests in `modules/00_foundations/tests.py`.

---

```
===================================================================================
 MODULE 01: Data Pipeline & Tokenization (From Words to Numbers)
===================================================================================
```
### 🎯 Learning Objectives
- Master tokenization paradigms: Character-level vs Word-level vs Subword-level.
- Implement Byte-Pair Encoding (BPE) from scratch.
- Understand special tokens (`<|bos|>`, `<|eos|>`, `<|pad|>`, `<|unk|>`).
- Build a robust PyTorch `Dataset` and `DataLoader` with causal target shifting.

### 🧠 Core Concepts & Theory
1. **The Tokenization Tradeoff:**
   - Character-level: Tiny vocabulary ($~256$), but sequences are very long (high attention compute $O(T^2)$).
   - Word-level: Out-of-vocabulary (OOV) nightmare, enormous embedding table.
   - Subword (BPE / WordPiece): The sweet spot balancing sequence length and vocabulary size.
2. **Byte-Pair Encoding (BPE) Mechanics:**
   - Byte-level representations (handling arbitrary UTF-8 without `<UNK>`).
   - Iterative frequency counting of adjacent pairs.
   - Merge operations and vocabulary construction.
3. **Data Batching & Causal Shifting:**
   - Context window ($T_{\text{ctx}}$), Batch size ($B$).
   - Input $x = [t_1, t_2, \dots, t_{T-1}]$ vs Target $y = [t_2, t_3, \dots, t_T]$.
   - Padding and attention masks for variable-length stories.

### 💻 Hands-on Lab: Custom StoryTokenizer & Data Pipeline
- **Files:** `modules/01_tokenization/bpe_tokenizer.py` & `dataset.py`
- Train a $4,096$-token BPE tokenizer directly on the TinyStories dataset.
- Implement `encode()`, `decode()`, and save/load serialization.
- Build a PyTorch `StoryDataset` that extracts random overlapping chunks of length $256$ for efficient GPU streaming.
- **Verification:** `tests.py` testing roundtrip `decode(encode(text)) == text` and tensor batch shapes.

---

```
===================================================================================
 MODULE 02: Vector Space & Positional Representations
===================================================================================
```
### 🎯 Learning Objectives
- Understand lookup tables and continuous semantic spaces.
- Discover why Transformer self-attention is permutation-invariant (order-blind).
- Implement Learned Absolute Positional Embeddings and modern Rotary Position Embeddings (RoPE).

### 🧠 Core Concepts & Theory
1. **Token Embeddings ($W_E \in \mathbb{R}^{V \times d_{\text{model}}}$):**
   - Matrix multiplication vs index lookup.
   - High-dimensional vector space geometry: dot product, cosine similarity.
2. **The Positional Problem:**
   - Self-attention treats sequences as sets: `"Dog bites boy"` vs `"Boy bites dog"` produce identical attention weights without position encoding.
3. **Evolution of Positional Encodings:**
   - *Learned Absolute (GPT-2 style)*: $W_{\text{pos}} \in \mathbb{R}^{T_{\text{max}} \times d}$, simple but cannot extrapolate beyond training context length.
   - *Rotary Position Embedding (RoPE - LLaMA/Mistral style)*: Encoding relative distances by rotating query and key vectors in complex 2D subspaces.
     $$\mathbf{q}_m = \mathbf{R}_{\Theta, m} W_q \mathbf{x}_m, \quad \mathbf{k}_n = \mathbf{R}_{\Theta, n} W_k \mathbf{x}_n \implies \mathbf{q}_m^T \mathbf{k}_n = g(\mathbf{x}_m, \mathbf{x}_n, m - n)$$

### 💻 Hands-on Lab: Vector Space & RoPE Implementation
- **Files:** `modules/02_embeddings/rope.py` & `embed_visualizer.py`
- Implement 2D coordinate rotation matrices and complex tensor rotation for RoPE.
- Prove numerically that the inner product between query at position $m$ and key at position $n$ depends solely on the relative offset $(m - n)$.
- Visualize token embeddings in 2D with PCA / t-SNE (inspecting story clusters: animals, colors, verbs).
- **Verification:** `tests.py` verifying relative shift invariance.

---

```
===================================================================================
 MODULE 03: The Modern Transformer Architecture from Scratch
===================================================================================
```
### 🎯 Learning Objectives
- Demystify Query, Key, Value ($Q, K, V$) projections.
- Implement Scaled Dot-Product Attention and Causal Masking.
- Build Multi-Head Attention (MHA) and Grouped-Query Attention (GQA).
- Implement modern sub-components: RMSNorm and SwiGLU Feed-Forward Networks.
- Assemble the complete `StoryCraftLM` (modern LLaMA-style decoder-only architecture).

### 🧠 Core Concepts & Theory
1. **Self-Attention as Soft Database Lookup:**
   - **Query:** "What information am I looking for?"
   - **Key:** "What information do I contain?"
   - **Value:** "If selected, what content do I communicate?"
   - Formula:
     $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
   - Why scale by $\frac{1}{\sqrt{d_k}}$? Preventing gradient vanishing caused by extreme softmax outputs in high dimensions.
2. **Causal Masking ($M$):**
   - Upper triangular matrix of $-\infty$ to enforce that token $t$ can only attend to tokens $\le t$.
3. **Multi-Head Attention (MHA) & Grouped-Query Attention (GQA):**
   - Splitting $d_{\text{model}}$ into $H$ heads so the model can attend to multiple concepts simultaneously.
   - GQA (LLaMA 3 / Mistral): Sharing Key/Value heads across multiple Query heads to drastically reduce KV memory during generation.
4. **Feed-Forward Layer (MLP):**
   - Traditional: $\text{GELU}(x W_1) W_2$.
   - Modern: SwiGLU ($\text{Swish}(x W_{\text{gate}}) \odot (x W_{\text{up}}) W_{\text{down}}$).
5. **Modern Normalization & Residual Connections:**
   - Pre-LN formulation for gradient highway stability.
   - LayerNorm vs **RMSNorm** (Root Mean Square Normalization—faster and mathematically cleaner).

### 💻 Hands-on Lab: Assembling StoryCraft-LLM
- **File:** `modules/03_transformer_architecture/model.py`
- Build modular components:
  - `RMSNorm(dim, eps)`
  - `RotaryEmbedding(dim, max_seq_len)`
  - `CausalSelfAttention(d_model, n_heads, n_kv_heads)`
  - `SwiGLUFeedForward(d_model, hidden_dim)`
  - `TransformerBlock(d_model, n_heads, hidden_dim)`
  - `StoryCraftLM(vocab_size, d_model, n_layers, n_heads)`
- Verify parameter count (~8.5 million parameters).
- **Verification:** `tests.py` testing causal masking integrity and gradient backpropagation through all layers.

---

```
===================================================================================
 MODULE 04: The Pre-training Engine & Optimization
===================================================================================
```
### 🎯 Learning Objectives
- Implement the complete pre-training loop in PyTorch.
- Master optimization dynamics: Cross-Entropy Loss, Perplexity, AdamW, and Cosine Annealing with Warmup.
- Leverage Mixed Precision (`torch.amp.autocast`), Gradient Accumulation, and `torch.compile`.

### 🧠 Core Concepts & Theory
1. **The Objective Function:**
   - Cross-Entropy Loss: $\mathcal{L} = -\frac{1}{T} \sum_{t=1}^T \log P(w_t \mid w_{<t})$.
   - **Perplexity (PPL):** $e^{\mathcal{L}}$, representing the effective branching factor / uncertainty of the model at each word.
2. **The Optimizer: AdamW:**
   - Why not pure SGD? Sparse gradients, momentum, adaptive per-parameter learning rates.
   - Weight decay decoupling (Loshchilov & Hutter): why L2 regularization breaks with adaptive momentum.
3. **Training Bells and Whistles (from Raschka Appendix D):**
   - **Gradient Accumulation:** Simulating a batch size of 128 or 256 on a Mac/GPU with only enough memory for batch size 16 or 32.
   - **Learning Rate Schedule:** Linear Warmup (500 steps) + Cosine Decay down to $10\%$.
   - **Gradient Clipping:** Max norm $1.0$ to stabilize updates.
   - **Kernel Fusion & Speed:** `torch.compile(model)` and PyTorch 2.0 SDPA (`torch.nn.functional.scaled_dot_product_attention`) for 30–50% speedups.

### 💻 Hands-on Lab: Pre-training StoryCraft on TinyStories
- **File:** `modules/04_pretraining/train.py`
- Train on 100,000 TinyStories samples.
- Log train loss, validation loss, perplexity, and learning rate curves.
- Periodic generation checkpoints: every 500 steps, prompt the model with `"Once upon a time, Lily"` and watch the text evolve from gibberish $\to$ partial words $\to$ coherent sentences!
- Save best checkpoints (`checkpoints/storycraft_base.pt`).

---

```
===================================================================================
 MODULE 05: Generation, Sampling & Foundation Model Weight Loading
===================================================================================
```
### 🎯 Learning Objectives
- Understand the autoregressive decoding cycle: logits $\to$ probabilities $\to$ token ID $\to$ append.
- Implement Greedy Decoding vs Stochastic Sampling (Temperature, Top-k, Top-p, Min-p).
- Implement repetition penalties to avoid degenerate loops.
- **Cross-Compatibility Lab:** Load open pretrained weights (e.g., GPT-2 or SmolLM / LLaMA-3.2) into our scratch architecture by mapping state dictionaries!

### 🧠 Core Concepts & Theory
1. **The Autoregressive Loop:**
   - Iterative sequence expansion: $x_{t+1} \sim P(\cdot \mid x_{1:t})$.
2. **Sampling Hyperparameters:**
   - **Temperature ($T$):** Softmax sharpness scaling $P(i) \propto \exp(z_i / T)$.
   - **Top-$k$:** Truncate to $k$ most likely candidates.
   - **Top-$p$ (Nucleus):** Dynamically truncate to smallest set with cumulative mass $\ge p$.
   - **Repetition Penalty:** Discounting previously seen tokens.
3. **Weight Mapping & Architecture Verification (Raschka Ch 05 Pattern):**
   - How are state dictionaries structured in production weights?
   - Mapping parameter names: `transformer.h.0.attn.c_attn.weight` $\to$ `blocks.0.attn.qkv_proj.weight`.
   - Transposing linear weights (e.g., Conv1D vs nn.Linear).
   - Proving that our scratch-built architecture can execute real foundation weights!

### 💻 Hands-on Lab: Generation CLI & Weight Importer
- **Files:** `modules/05_generation_sampling/generate.py` & `load_pretrained.py`
- Build interactive story player with sampling sliders.
- Implement state-dict conversion script to load external open weights into our `StoryCraftLM`.

---

```
===================================================================================
 MODULE 06: Discriminative LLMs - Fine-Tuning for Story Classification
===================================================================================
```
### 🎯 Learning Objectives
- Understand the dual-use nature of LLMs: Generative vs Discriminative tasks.
- Modify the language model architecture for sequence classification.
- Fine-tune `StoryCraftLM` into a multi-task **Story Genre, Safety & Age-Appropriateness Classifier**.

### 🧠 Core Concepts & Theory (Inspired by Raschka Ch 06)
1. **Adapting a Pre-trained LLM for Classification:**
   - In generative pre-training, the model predicts $V$ tokens at each position $t$.
   - In sequence classification, we want a single prediction for the entire story:
     - Replace the LM Head ($d_{\text{model}} \to V$) with a Classification Head ($d_{\text{model}} \to K$ classes).
     - Which token embedding do we feed into the classification head?
       - Option A: Mean pooling across all tokens.
       - Option B: Last token representation (since causal attention ensures the final token has attended to the entire story!).
2. **Story Moderation & Classification Tasks:**
   - **Task 1 (Genre):** Adventure, Bedtime, Animal, Fantasy.
   - **Task 2 (Target Age):** Toddler (2-3), Preschool (4-5), Early Reader (6+).
   - **Task 3 (Content Safety):** Safe vs Unsafe/Scary.
3. **Freezing vs Full Fine-Tuning:**
   - Freezing Transformer backbone weights and training only the classification head vs fine-tuning the last 2 layers vs full-model tuning.

### 💻 Hands-on Lab: Building the Story Moderation Classifier
- **File:** `modules/06_classification/train_classifier.py`
- Build `StoryClassifier(backbone, num_classes)`.
- Train on 5,000 labeled stories.
- Evaluate Accuracy, Precision, Recall, and Confusion Matrix on held-out stories.

---

```
===================================================================================
 MODULE 07: Post-Training Phase 1 - Instruction Tuning (SFT) & LoRA from Scratch
===================================================================================
```
### 🎯 Learning Objectives
- Transform the base text completer into an instruction-following assistant.
- Format dialogues with ChatML templates and implement target-only loss masking.
- **Implement LoRA (Low-Rank Adaptation) from scratch** in pure PyTorch (inspired by Raschka Appendix E).
- Train modular Story LoRA Adapters (e.g., Bedtime LoRA vs Mystery LoRA) updating $< 1\%$ of parameters.

### 🧠 Core Concepts & Theory
1. **Base Model vs Instruct Model:**
   - Base models continue text; instruct models follow instructions.
2. **ChatML Format & Loss Masking:**
   - Formatting System, User, and Assistant turns.
   - Masking user prompt tokens with `ignore_index = -100` so loss is only computed on the assistant story.
3. **Parameter-Efficient Fine-Tuning: LoRA from Scratch:**
   - Full fine-tuning requires saving all 8.5M weights for every task.
   - LoRA (Hu et al., 2021) freezes pre-trained weight $W_0 \in \mathbb{R}^{d \times k}$ and adds a low-rank decomposition:
     $$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
     where $A \in \mathbb{R}^{r \times k}$ (initialized with Gaussian noise) and $B \in \mathbb{R}^{d \times r}$ (initialized with zeros).
   - With rank $r = 4$, parameter count drops from millions to **~45,000 trainable weights**!
   - Zero inference latency overhead: $W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A$.

### 💻 Hands-on Lab: Building LoRALinear & SFT Training
- **Files:** `modules/07_instruction_tuning_sft/lora.py` & `train_sft.py`
- Code `LoRALinear(in_features, out_features, rank, alpha)` from scratch.
- Replace attention projection layers with LoRA layers.
- Fine-tune base model on 10,000 instruction-story pairs to produce `storycraft_instruct_lora.pt`.

---

```
===================================================================================
 MODULE 08: Post-Training Phase 2 - Alignment via DPO
===================================================================================
```
### 🎯 Learning Objectives
- Understand the alignment problem: why supervised fine-tuning cannot penalize negative behaviors effectively.
- Contrast RLHF (PPO) with Direct Preference Optimization (DPO).
- Construct pairwise preference datasets: `(prompt, chosen_story, rejected_story)`.
- Train `storycraft_aligned.pt` using the DPO loss function.

### 🧠 Core Concepts & Theory (Raschka Ch 07.04 Pattern)
1. **Why Supervised Fine-Tuning is Insufficient:**
   - SFT provides no gradient signal for what *not* to say (e.g., scary tropes, abrupt endings, rude characters).
2. **From RLHF to DPO:**
   - DPO (Rafailov et al., 2023) proves the language model policy implicitly defines its own reward:
     $$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$
   - Increases probability of winning story $y_w$ relative to frozen reference model $\pi_{\text{ref}}$, while decreasing probability of losing story $y_l$.

### 💻 Hands-on Lab: Aligning StoryCraft with DPO
- **File:** `modules/08_alignment_dpo/train_dpo.py`
- Build preference dataset: wholesome endings ($y_w$) vs unresolved/scary endings ($y_l$).
- Implement DPO loss and optimize policy model.
- Side-by-side comparison: Instruct vs DPO responses.

---

```
===================================================================================
 MODULE 09: Evaluation, Benchmarking & Safety
===================================================================================
```
### 🎯 Learning Objectives
- Quantify model capabilities beyond simple training loss.
- Implement intrinsic evaluations (Perplexity on held-out stories).
- Build an LLM-as-a-Judge automated scoring harness using external APIs (Gemini/GPT) or local models (Ollama).
- Test adversarial safety and robustness.

### 🧠 Core Concepts & Theory
1. **Evaluation Dimensions:**
   - *Intrinsic:* Perplexity, bits-per-token.
   - *Linguistic Diversity:* Distinct-1 and Distinct-2 n-gram ratios.
   - *Qualitative Rubrics:* Plot coherence, character consistency, age appropriateness.
2. **LLM-as-a-Judge Automation:**
   - Using larger models to evaluate student models based on structured JSON rubrics.
3. **Safety Probing:**
   - Jailbreak and bias testing on children's stories.

### 💻 Hands-on Lab: The StoryBenchmark Suite
- **File:** `modules/09_evaluation/benchmark.py` & `llm_judge.py`
- Run automated benchmarks across 100 test prompts.
- Generate automated evaluation report `benchmark_report.json`.

---

```
===================================================================================
 MODULE 10: Inference Optimization & Key-Value Caching
===================================================================================
```
### 🎯 Learning Objectives
- Understand the quadratic computational inefficiency of naive autoregressive inference.
- Implement a Key-Value (KV) Cache from scratch inside the attention blocks.
- Explore Post-Training Quantization (FP32 $\to$ FP16 $\to$ INT8 / GGUF).

### 🧠 Core Concepts & Theory
1. **The Inference Bottleneck:**
   - Naive inference re-computes $K$ and $V$ for all past tokens at every new step ($O(T^2)$ operations).
2. **KV Cache Mechanics:**
   - Cache historical Key and Value tensors.
   - At step $t$, compute $q_t, k_t, v_t$ only for the newest token, append $k_t, v_t$ to cache, and compute attention in $O(1)$ per step!
3. **Quantization:**
   - Reducing weight precision to INT8/INT4 for faster CPU/edge throughput.

### 💻 Hands-on Lab: Fast Story Streaming Engine
- **File:** `modules/10_kv_cache_optimization/kv_cache_model.py`
- Benchmark TPS (Tokens Per Second) with vs without KV Cache.
- Demonstrate $> 3.5\times$ speedup on 256-token story generations.

---

```
===================================================================================
 MODULE 11: Capstone Project - The Interactive Storybook App
===================================================================================
```
### 🎯 Learning Objectives
- Package the trained model into a production-ready application.
- Build a streaming FastAPI backend and a clean web UI for real-time storytelling.
- Add advanced generation features: story branching ("Choose-Your-Own-Adventure") and character persistence.

### 💻 Hands-on Lab: The StoryCraft Studio Application
- **Directory:** `modules/11_capstone_storybook_app/`
- **Backend:** `server.py` (FastAPI with Server-Sent Events token streamer).
- **Frontend:** `web/index.html` (Modern UI with typewriter effect, genre picker, moral selector, and story branching).

---

## 🔬 Appendices: Advanced Topics & Deeper Dives

Inspired by the supplementary research in `LLMs-from-scratch`:

- **Appendix A: Distributed Data Parallel (DDP) & Multi-GPU Training**
  - Multi-GPU scaling, NCCL Ring All-Reduce, gradient synchronization, and launch scripts with `torchrun`.
- **Appendix B: Frontier Attention Architectures (GQA, MLA, SWA, MoE)**
  - Grouped-Query Attention (GQA), DeepSeek-style Multi-head Latent Attention (MLA), Mistral Sliding Window Attention (SWA), and Mixture of Experts (MoE).
- **Appendix C: Cross-Architecture Weight Importers**
  - Scripts for loading weights from OpenAI GPT-2, LLaMA-3.2-1B, and SmolLM-135M directly into scratch-built architectures.
- **Appendix D: Training Speedups & Modern Optimizers**
  - FlashAttention-2, `torch.compile`, and the Muon optimizer.
