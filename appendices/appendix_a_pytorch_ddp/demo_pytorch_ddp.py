import os
import sys
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)

from storycraft import set_seed


def simulate_ddp_allreduce(num_workers: int = 4):
    print("")
    print("1. Simulating Multi-GPU DistributedDataParallel (DDP):")
    print("   - Each GPU runs as an independent process via `torchrun` (No Python GIL bottleneck).")
    print("   - Workers process different slices of the training batch in parallel.")
    print(f"   - Cluster Size: {num_workers} simulated GPU workers.")

    weights = torch.tensor([1.0, 2.0, -1.0])

    local_losses = []
    local_grads = []
    print("")
    print("2. Local Forward & Backward Computations on Each GPU Worker:")
    for rank in range(num_workers):
        local_x = torch.randn(3)
        local_loss = torch.sum((weights - local_x) ** 2)
        local_grad = 2 * (weights - local_x)
        local_losses.append(local_loss.item())
        local_grads.append(local_grad)
        print(f"   [Worker Rank {rank}] Local Loss: {local_loss.item():.4f} | Local Gradient: {local_grad.tolist()}")

    print("")
    print("3. Executing NCCL Ring All-Reduce Gradient Synchronization:")
    synced_grad = torch.stack(local_grads).mean(dim=0)
    print(f"   ✓ Synchronized Global Gradient: {synced_grad.tolist()}")

    lr = 0.05
    new_weights = weights - lr * synced_grad
    print("")
    print("4. Simultaneous Weight Update in Lockstep:")
    print(f"   - Initial Weights:  {weights.tolist()}")
    print(f"   - Updated Weights:  {new_weights.tolist()}")
    print("   ✓ All workers remain perfectly synchronized without a central master bottleneck!")


def main():
    set_seed(42)
    print("=" * 70)
    print("🛠️ APPENDIX A: DISTRIBUTED DATA PARALLEL (DDP) SIMULATION")
    print("=" * 70)

    simulate_ddp_allreduce(num_workers=4)

    print("")
    print("=" * 70)
    print("💡 KEY PEDAGOGICAL TAKEAWAYS:")
    print("=" * 70)
    print("1. DDP launches independent processes per GPU, avoiding Python GIL locks.")
    print("2. Gradients are overlapped and averaged via Ring All-Reduce during backward pass.")
    print("3. For the foundational PyTorch Primer, see: pytorch_primer/demo_pytorch_primer.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
