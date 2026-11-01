# 🔤 Module 01: Data Pipeline & Tokenization

> **Central Anchor in this Module:** Training a Byte-Pair Encoding (BPE) tokenizer specifically on children's stories, constructing vocabulary dictionaries, and packaging story text into a PyTorch DataLoader.

---

## 🎯 Learning Objectives
1. Understand the tradeoffs between Character-level, Word-level, and Subword-level tokenization.
2. Build a Byte-Pair Encoding (BPE) tokenizer from scratch.
3. Understand special tokens (`<|bos|>`, `<|eos|>`, `<|pad|>`, `<|unk|>`).
4. Build a PyTorch `StoryDataset` that creates sliding causal training chunks with shifted targets:
   $$x = [t_1, t_2, \dots, t_{T-1}], \quad y = [t_2, t_3, \dots, t_T]$$

---

## 🔬 Core Theory: Byte-Pair Encoding (BPE)
Computers cannot read text directly; they process numbers (token IDs).

### The BPE Algorithm:
1. Start with the individual byte representations of all characters in the training corpus (vocabulary size = 256).
2. Count the frequency of every consecutive pair of tokens across the corpus.
3. Find the most frequent pair (e.g., `('t', 'h')` $\to$ `'th'`).
4. Merge that pair into a new single token and append it to the vocabulary.
5. Repeat steps 2–4 until reaching the target vocabulary size $V$ (e.g., $V = 4,096$).

### Causal Batching:
For autoregressive generation, we train the model to predict the *next* token at every position:
```
Sequence:  ["Once", "upon", "a", "time", "there", "was"]
Input (x): ["Once", "upon", "a", "time", "there"]
Target(y): ["upon", "a", "time", "there", "was"]
```

---

## 💻 Lab Exercises
1. `bpe_tokenizer.py`:
   - Implement `train(text, vocab_size=4096)`
   - Implement `encode(text) -> list[int]`
   - Implement `decode(tokens) -> str`
2. `dataset.py`:
   - Implement `StoryDataset(tokens, max_seq_len=256)`
   - Create PyTorch `DataLoader(batch_size=32, shuffle=True)`
