# 🔄 Appendix C: Cross-Architecture Weight Importers

> **Central Anchor in this Appendix:** Bridging our scratch-built `StoryCraftLM` architecture with industry weights by mapping parameter dictionaries from open foundation models (e.g. GPT-2, LLaMA 3.2, SmolLM).

---

## 🎯 Learning Objectives
1. Understand how PyTorch `state_dict` objects serialize model weights.
2. Translate naming conventions between HuggingFace Transformers and custom architectures.
3. Handle tensor transpositions (`nn.Linear` vs Conv1D) and composite weight splitting (e.g., combined `qkv_proj` to split `q`, `k`, `v`).
4. Prove that our scratch-built code is 100% mathematically compatible with real foundation models!

---

## 🔬 Core Theory: State-Dict Mapping Table

| HuggingFace GPT-2 Key | `StoryCraftLM` Key | Transformation |
| :--- | :--- | :--- |
| `transformer.wte.weight` | `tok_embed.weight` | Direct copy |
| `transformer.h.<i>.ln_1.weight` | `blocks.<i>.norm1.weight` | Direct copy |
| `transformer.h.<i>.attn.c_attn.weight` | `blocks.<i>.attn.q_proj.weight`, `k_proj`, `v_proj` | Split into 3 chunks along dim 0 & transpose |
| `transformer.h.<i>.mlp.c_fc.weight` | `blocks.<i>.mlp.gate_proj.weight` | Transpose |
| `transformer.ln_f.weight` | `norm_final.weight` | Direct copy |

---

## 💻 Included Demonstration Script
Run the companion demo:
```bash
python3 appendices/appendix_c_weight_loading/demo_weight_loading.py
```
This script demonstrates:
- Generating a mock Hugging Face model state dict.
- Running an automated translation function that remaps parameter tensors.
- Loading the weights into our `StoryCraftLM` architecture and verifying matching logits.
