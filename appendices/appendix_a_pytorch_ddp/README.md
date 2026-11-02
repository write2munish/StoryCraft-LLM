# 🛠️ Appendix A: PyTorch Primer & Distributed Data Parallel (DDP)

> **Central Anchor in this Appendix:** Mastering low-level tensor mechanics, autograd graphs, and scaling `StoryCraftLM` training across multiple GPUs using DistributedDataParallel (DDP).

---

## 🎯 Learning Objectives
1. Understand PyTorch storage, strides, views, and memory management across devices (`cpu`, `mps`, `cuda`).
2. Demystify the Autograd graph: dynamic graph construction, backward execution, and gradient accumulation.
3. Understand why `DistributedDataParallel` (DDP) is superior to legacy `DataParallel` (DP).
4. Learn how `torchrun` coordinates multi-process distributed training and synchronizes gradients via Ring All-Reduce.

---

## 🔬 Core Theory: From Single-Device to Multi-GPU

### 1. DataParallel (DP) vs. DistributedDataParallel (DDP)
- **DataParallel (Single Process, Multi-Thread):**
  - Runs in one Python process subject to the Global Interpreter Lock (GIL).
  - Gathers all outputs and gradients back to GPU 0 (creating a severe communication bottleneck on GPU 0).
- **DistributedDataParallel (Multi-Process):**
  - Launches 1 independent Python process per GPU (`torchrun`).
  - No GIL contention.
  - Synchronizes gradients during the backward pass in the background using the NCCL backend via Ring All-Reduce.

### 2. Standard DDP Launch Template
```bash
torchrun --standalone --nproc_per_node=4 train_ddp.py
```

---

## 💻 Included Demonstration Script
Run the companion demo:
```bash
python3 appendices/appendix_a_pytorch_ddp/demo_pytorch_ddp.py
```
This script demonstrates:
- PyTorch tensor memory mechanics (strides and contiguous tensors).
- Autograd graph inspection (`grad_fn` chaining).
- A pure-Python simulation of multi-worker All-Reduce gradient synchronization.
