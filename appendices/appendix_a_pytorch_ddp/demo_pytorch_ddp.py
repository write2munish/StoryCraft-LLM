"""
Appendix A: PyTorch Essentials & Distributed Data Parallel (DDP) Simulation
=============================================================================
Demonstrates:
1. Tensor memory layouts (strides, views, and contiguous buffers).
2. Autograd graph tracking and gradient backpropagation.
3. Multi-GPU Distributed Data Parallel (DDP) gradient synchronization simulation.

Run this script:
    python3 appendices/appendix_a_pytorch_ddp/demo_pytorch_ddp.py
"""

import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)
from storycraft import set_seed

import torch
import torch.nn as nn


def demonstrate_tensor_mechanics():
    print("\n1. Tensor Memory Layouts & Strides:")
    t = torch.arange(12).reshape(3, 4)
    print(f"   Tensor Shape: {t.shape}, Strides: {t.stride()}")
    print("   Transpose (view without copy):")
    t_trans = t.t()
    print(f"   Transposed Shape: {t_trans.shape}, Strides: {t_trans.stride()}, Contiguous: {t_trans.is_contiguous()}")
    t_contig = t_trans.contiguous()
    print(f"   After .contiguous() -> Strides: {t_contig.stride()}, Contiguous: {t_contig.is_contiguous()}")


def demonstrate_autograd():
    print("\n2. Autograd Computation Graph:")
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    # y = x^2 + 5x
    y = x ** 2 + 5 * x
    loss = y.sum()
    print(f"   Forward: x={x.tolist()} -> y={y.tolist()} -> loss={loss.item():.2f}")
    print(f"   Gradient Function Chain: loss.grad_fn = {loss.grad_fn}")

    loss.backward()
    print(f"   Analytic Gradient dy/dx = 2x + 5: expected [9.0, 11.0]")
    print(f"   Computed x.grad: {x.grad.tolist()}")


def simulate_ddp_allreduce():
    print("\n3. Simulating Multi-GPU DistributedDataParallel (DDP):")
    print("   In DDP, each GPU computes local batch gradients.")
    print("   Before updating weights, NCCL All-Reduce averages gradients across all workers!")

    num_workers = 3
    torch.manual_seed(42)

    # Simulated model weights (identical across all workers at start)
    weights = torch.tensor([1.0, 2.0, -1.0])

    # Each worker receives a different local batch of training data
    local_losses = []
    local_grads = []
    for rank in range(num_workers):
        local_x = torch.randn(3)
        # local loss = sum((weights - local_x)^2)
        local_loss = torch.sum((weights - local_x) ** 2)
        local_grad = 2 * (weights - local_x)
        local_losses.append(local_loss.item())
        local_grads.append(local_grad)
        print(f"   [Worker {rank}] Local Loss: {local_loss.item():.4f}, Local Grad: {local_grad.tolist()}")

    # Ring All-Reduce step (averaging gradients)
    synced_grad = torch.stack(local_grads).mean(dim=0)
    print(f"\n   --- All-Reduce Gradient Synchronization ---")
    print(f"   Synchronized Global Gradient: {synced_grad.tolist()}")

    # Weight update on all workers simultaneously
    lr = 0.05
    new_weights = weights - lr * synced_grad
    print(f"   Updated Model Weights:        {new_weights.tolist()}")
    print("   All workers remain perfectly in sync without master-worker bottleneck!")


def main():
    set_seed(42)
    print("=" * 70)
    print("🛠️ APPENDIX A: PYTORCH PRIMER & DISTRIBUTED DATA PARALLEL (DDP)")
    print("=" * 70)

    demonstrate_tensor_mechanics()
    demonstrate_autograd()
    simulate_ddp_allreduce()

    print("\n" + "=" * 70)
    print("💡 KEY PEDAGOGICAL TAKEAWAYS:")
    print("=" * 70)
    print("1. DDP launches independent processes per GPU, avoiding Python GIL locks.")
    print("2. Gradients are overlapped and averaged via Ring All-Reduce during backward pass.")
    print("3. Memory contiguousness is required for high-throughput CUDA GEMM kernels.")
    print("=" * 70)


if __name__ == "__main__":
    main()
