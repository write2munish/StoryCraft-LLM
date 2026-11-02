# 🔤 Module 01: Data Pipeline & Tokenization

> **Central Anchor:** Training a Byte-Pair Encoding (BPE) subword tokenizer specifically on children's stories, building vocabulary mappings, and packaging text into PyTorch causal training batches.

---

## 🎯 What to Expect
- **What You Will Build & Run:** You will execute `demo_tokenization.py`, which implements an end-to-end Byte-Pair Encoding (BPE) algorithm from scratch and formats text into PyTorch training tensors.
- **What You Will See:** The terminal displays the iterative merge process (e.g. merging `'l'` + `'i'` $\to$ `'li'`, then `'li'` + `'ly'` $\to$ `'lily'`), encodes sample story sentences into integer token IDs, decodes them back to English, and displays causal input-target training pairs ($x$ vs $y$).
- **What You Will Observe:** You will see how subword tokenization prevents out-of-vocabulary (`<UNK>`) errors while compressing raw character streams into compact sequences.
- **Run Command:**
  ```bash
  python3 modules/01_tokenization/demo_tokenization.py
  ```

---

## 💡 The Concept Explained Simply
Computers cannot read letters, words, or sentences—they can only do math on numbers. But how should we translate human text into numbers?
- **Option 1: Character-by-character (`'l'`, `'i'`, `'l'`, `'y'`).** The vocabulary is tiny (26 letters), but words become long sequences. A 300-word children's story turns into 1,800 characters, making attention calculations extremely slow and expensive.
- **Option 2: Whole words (`"lily"`, `"played"`, `"garden"`).** The sequences are short, but the vocabulary explodes. If the model sees `"baked"` and `"baking"`, it treats them as completely separate words. If a child enters a typo or rare word, the model crashes with an Unknown Token (`<UNK>`) error!
- **Option 3: The Goldilocks Solution — Byte-Pair Encoding (BPE).** BPE starts with individual letters and iteratively merges the most frequent pairs into subwords. Common words get their own ID (`"dragon"` $\to$ `42`), while rare words are broken into understandable pieces (`"un"` + `"friendly"` $\to$ `[104, 88]`). No word is ever unknown, and sequence lengths stay short!

---

## 🔬 Core Theory: Byte-Pair Encoding (BPE)
BPE constructs a subword vocabulary through frequency-driven iterative compression:
1. Initialize the vocabulary with all unique base bytes (characters) in the training text.
2. Count the frequency of every adjacent symbol pair across the corpus.
3. Identify the most frequent pair $(c_1, c_2)$ and merge it into a new composite symbol $c_{12}$.
4. Add $c_{12}$ to the vocabulary and replace all occurrences of $(c_1, c_2)$ in the corpus.
5. Repeat steps 2–4 until reaching the target vocabulary size $V$ (e.g., $V = 256$ in StoryCraft).

### Causal Shifted Pairs ($x$ vs $y$):
Autoregressive training is self-supervised. We construct input $x$ and target $y$ by shifting the sequence by exactly 1 position:
$$x = [t_1, t_2, \dots, t_{T-1}], \quad y = [t_2, t_3, \dots, t_T]$$
At position $t$, the model takes tokens $x_{1:t}$ and is trained to predict $y_t = x_{t+1}$.

---

## 💻 File Structure & Runnable Components
```
modules/01_tokenization/
├── README.md               # This module guide
├── demo_tokenization.py    # Runnable BPE tokenizer & PyTorch DataLoader demo
└── tests.py                # 4-test unit suite verifying merges and encode/decode
```

---

## 🔑 Key Pedagogical Takeaways
1. **Subword Sweet Spot:** BPE eliminates out-of-vocabulary errors while compressing text into manageable sequence lengths.
2. **Control Tokens:** Special tokens like `<|bos|>`, `<|eos|>`, and `<|pad|>` provide structural boundaries for prompts, generation termination, and batch alignment.
3. **Self-Supervision via Shifting:** Next-token prediction requires no manual labels; shifting token sequences by 1 creates ground-truth training targets automatically.
4. **Bridge to Module 02:** Discrete token integers (`[43, 108]`) cannot be multiplied or passed through neural layers; Module 02 maps them into continuous vector geometry.

---

## 🧪 Running the Code & Unit Tests
```bash
# Run the interactive demonstration
python3 modules/01_tokenization/demo_tokenization.py

# Run the automated unit tests
python3 modules/01_tokenization/tests.py
```
