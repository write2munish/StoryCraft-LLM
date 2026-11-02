# 🚀 Front-Loaded Prerequisite: PyTorch Primer for LLM Architecture

> **Front-Loaded Prerequisite:** Mastering tensor dimensions, memory strides, broadcasting rules, autograd mechanics, and GPU acceleration *before* diving into language modeling.

---

## 🎯 What to Expect
- **What You Will Build & Run:** You will execute `demo_pytorch_primer.py`, a comprehensive, runnable primer that demystifies how PyTorch actually works under the hood.
- **What You Will See:** The terminal displays tensor memory layouts (strides, views, contiguous memory), demonstrates the exact broadcasting mechanics used in Multi-Head Attention and RoPE, traces the dynamic computation graph during backward passes, and detects available hardware accelerators (`mps`, `cuda`, `cpu`).
- **What You Will Learn:** You will gain the exact mathematical and systems engineering confidence required to understand every line of PyTorch code in Modules 01 through 11 without getting tripped up by shape mismatches or memory errors.
- **Run Command:**
  ```bash
  python3 pytorch_primer/demo_pytorch_primer.py
  ```

---

## 💡 The Concept Explained Simply
If an LLM is a high-performance race car, **PyTorch is the engine machine shop**. 

Why can't we just write an LLM using standard Python lists and for-loops?
1. **Python Loops are 100x Too Slow:** Doing a matrix multiplication in pure Python on $1,000 	imes 1,000$ numbers takes several seconds. In PyTorch, optimized C++ and CUDA kernels run it in *microseconds* because they process thousands of numbers in parallel across GPU cores.
2. **What is a Tensor?** A tensor is simply a multi-dimensional array of numbers with a specific data type (e.g. 32-bit float) and an address in memory (RAM or VRAM). A vector is 1D, a matrix is 2D, and an LLM batch is 4D: `[Batch_Size, Num_Heads, Seq_Length, Head_Dim]`.
3. **The Secret of Views & Strides:** When you transpose a tensor in PyTorch (e.g. swapping sequence length and attention heads), PyTorch doesn't copy all those millions of numbers! It simply changes its "stride"—the mathematical rule for how to read the array. But when you later reshape that tensor, PyTorch might demand `.contiguous()`, which forces it to copy the numbers into fresh, contiguous memory.
4. **Broadcasting (Automatic Dimension Expansion):** How does a 1D position vector of length $T$ add to a 4D tensor `[B, H, T, D]`? PyTorch's **broadcasting rules** automatically stretch singleton dimensions without copying any memory!
5. **Autograd (The Calculus Engine):** In traditional software, you write code to compute outputs. In deep learning, you also need the *derivatives* of those outputs with respect to every weight. PyTorch tracks every operation you perform in a dynamic "DAG" (Directed Acyclic Graph) during the forward pass. Calling `.backward()` automatically runs the chain rule of calculus in reverse, delivering exact gradients to all weights instantly!

---

## 🔬 Core Theory: The Five Pillars of PyTorch for LLMs

### 1. Tensor Memory Layout, Storage & Strides
A PyTorch tensor consists of a 1D contiguous block of raw memory called a `Storage`, paired with metadata:
- **Shape:** The virtual multi-dimensional grid size (e.g. `(3, 4)`).
- **Stride:** The number of memory elements to skip in storage to move to the next item along each dimension.
For a 2D tensor of shape $(R, C)$, row-major stride is $(C, 1)$. Transposition swaps the stride tuple $(1, C)$ without moving data:
$$\text{is\_contiguous}() = \text{False}$$
Calling `.contiguous()` reallocates storage in standard memory order.

### 2. Broadcasting Semantics
Two tensors can be combined elementwise if, starting from trailing dimensions:
1. Dimensions are equal, OR
2. One of the dimensions is $1$.
Example in `StoryCraftLM` RoPE:
```
Query Tensor:    [Batch=2, Heads=4, Seq_Len=16, Head_Dim=32]
RoPE Cos/Sin:    [      1,       1, Seq_Len=16, Head_Dim=32]
Broadcast Out:   [Batch=2, Heads=4, Seq_Len=16, Head_Dim=32]  (Zero memory duplication!)
```

### 3. The Dynamic Autograd Graph
When `requires_grad=True`, operations create a computation graph:
$$x \xrightarrow{\text{PowBackward}} y = x^2 \xrightarrow{\text{AddBackward}} z = y + 5x \xrightarrow{\text{SumBackward}} \mathcal{L}$$
Calling `\mathcal{L}.backward()` traverses this graph backwards via reverse-mode automatic differentiation:
$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial z} \cdot \left( 2x + 5 \right)$$

### 4. Hardware Agnostic Device Placement
```python
device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
```

### 5. `nn.Module` Architecture & Parameter Registration
- Subclassing `nn.Module` automatically tracks trainable weights in `self.parameters()`.
- Non-trainable persistent states (like causal attention masks and RoPE frequency tables) are registered using `self.register_buffer(name, tensor)`, ensuring they automatically move with the model across devices.

---

## 💻 File Structure & Runnable Components
```
pytorch_primer/
├── README.md               # This front-loaded prerequisite guide
├── demo_pytorch_primer.py  # Interactive runnable tensor, autograd & hardware demo
└── tests.py                # 5-test unit suite verifying tensor mechanics & autograd
```

---

## 🔑 Key Pedagogical Takeaways
1. **Front-Loaded for Success:** Understanding strides, broadcasting, and autograd *first* ensures you never get confused by tensor manipulation in the core LLM modules.
2. **Zero-Copy Views:** Tensor slicing, reshaping, and transpositions avoid memory allocation when possible; call `.contiguous()` when memory rearrangement is required.
3. **Broadcasting Powers Attention:** Broadcasting enables applying rotary positional shifts and attention masks across batches and heads with maximum cache efficiency.
4. **Autograd Manages Calculus:** The dynamic computation graph automates backpropagation; use `torch.no_grad()` during inference to disable graph building and save memory.
5. **Bridge to Module 00:** With PyTorch fundamentals locked down, you are ready to begin **Module 00: Foundations of Language Modeling**!

---

## 🧪 Running the Code & Unit Tests
```bash
# Run the interactive PyTorch Primer demo
python3 pytorch_primer/demo_pytorch_primer.py

# Run the automated unit tests
python3 pytorch_primer/tests.py
```
