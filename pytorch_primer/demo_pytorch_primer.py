import os
import sys
import torch
import torch.nn as nn

# Robust repo root discovery whether called directly or via symlink
cur = os.path.abspath(os.path.dirname(__file__))
repo_root = cur
while repo_root and repo_root != "/":
    if os.path.exists(os.path.join(repo_root, "storycraft")):
        break
    repo_root = os.path.dirname(repo_root)

sys.path.insert(0, repo_root)

from storycraft import set_seed


def demo_tensor_creation_and_devices():
    print("")
    print("1. Tensor Creation & Hardware Detection:")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon GPU (Metal Performance Shaders - MPS)"
    else:
        device = torch.device("cpu")
        device_name = "Host CPU"

    print(f"   ✓ Active Compute Device: {device} [{device_name}]")

    t_fp32 = torch.tensor([1.0, 2.5, 4.0], dtype=torch.float32, device=device)
    t_int64 = torch.tensor([104, 892, 334], dtype=torch.long, device=device)
    print(f"   - Float32 Tensor: {t_fp32} (shape: {t_fp32.shape}, dtype: {t_fp32.dtype})")
    print(f"   - Token IDs (Long): {t_int64} (shape: {t_int64.shape}, dtype: {t_int64.dtype})")


def demo_memory_strides_and_views():
    print("")
    print("2. Memory Layouts, Strides & Zero-Copy Views:")
    print("   A tensor is a 1D contiguous storage array viewed through 'strides'.")

    matrix = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print(f"   Original Matrix (3x4):\n{matrix}")
    print(f"   - Shape:   {matrix.shape}")
    print(f"   - Strides: {matrix.stride()}  (Skip 4 floats to move 1 row down; 1 float to move 1 col right)")
    print(f"   - Contiguous: {matrix.is_contiguous()}")

    transposed = matrix.t()
    print(f"\n   Transposed Matrix (4x3) [Zero-Copy View]:")
    print(f"   - Shape:   {transposed.shape}")
    print(f"   - Strides: {transposed.stride()}  (Strides swapped: (1, 4) instead of (4, 1))")
    print(f"   - Contiguous: {transposed.is_contiguous()}  (Notice: Non-contiguous in physical RAM!)")

    rearranged = transposed.contiguous().view(2, 6)
    print(f"\n   After .contiguous().view(2, 6):")
    print(f"   - Reshaped: {rearranged.shape}, Contiguous: {rearranged.is_contiguous()}")
    print("   💡 This is why Multi-Head Attention uses: out.transpose(1, 2).contiguous().view(B, T, C)!")


def demo_broadcasting_mechanics():
    print("")
    print("3. Broadcasting Rules (How RoPE & Attention Combine Dimensions):")
    print("   PyTorch automatically expands dimensions of size 1 without copying data.")

    query = torch.ones(2, 4, 8, 16)
    rope_cache = torch.full((1, 1, 8, 16), 2.0)
    product = query * rope_cache

    print(f"   - Query Shape:      {list(query.shape)}")
    print(f"   - RoPE Cache Shape: {list(rope_cache.shape)}")
    print(f"   - Output Shape:     {list(product.shape)}  (Broadcasting expanded dimensions 0 and 1 with 0 bytes overhead!)")
    print(f"   - Value Check:      Sample output element = {product[0, 0, 0, 0].item():.1f} (1.0 * 2.0)")


def demo_autograd_engine():
    print("")
    print("4. The Autograd Engine (Automatic Reverse-Mode Differentiation):")
    print("   PyTorch builds a dynamic computation graph during the forward pass.")

    x = torch.tensor([2.0, 4.0], requires_grad=True)
    y = x ** 2 + 5 * x
    loss = y.sum()

    print(f"   - Input x:     {x.tolist()}")
    print(f"   - Forward y:   {y.tolist()}")
    print(f"   - Scalar Loss: {loss.item():.2f}")
    print(f"   - Graph Node:  loss.grad_fn = {loss.grad_fn}")

    loss.backward()

    print(f"\n   --- Gradient Verification ---")
    print(f"   - Analytical dy/dx: expected [9.0, 13.0]")
    print(f"   - Computed x.grad:  {x.grad.tolist()} (Exact match!)")

    with torch.no_grad():
        inference_out = x * 10
        print(f"   - Under torch.no_grad(): requires_grad={inference_out.requires_grad} (Saves RAM during generation)")


def demo_custom_nn_module():
    print("")
    print("5. Custom nn.Module Architecture & Parameter Registration:")

    class MinimalMLP(nn.Module):
        def __init__(self, in_features=16, hidden_features=32):
            super().__init__()
            self.linear1 = nn.Linear(in_features, hidden_features)
            self.linear2 = nn.Linear(hidden_features, in_features)
            self.register_buffer("running_scale", torch.tensor([1.5]))

        def forward(self, x):
            return self.linear2(torch.relu(self.linear1(x))) * self.running_scale

    model = MinimalMLP(in_features=8, hidden_features=16)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   - Total Parameters:     {total_params}")
    print(f"   - Trainable Parameters: {trainable_params}")
    print(f"   - Registered Buffers:   {[name for name, _ in model.named_buffers()]}")

    dummy_input = torch.randn(2, 8)
    output = model(dummy_input)
    print(f"   - Forward Pass Shape:   {dummy_input.shape} -> {output.shape}")


def main():
    set_seed(42)
    print("=" * 70)
    print("🚀 FRONT-LOADED PREREQUISITE: PYTORCH PRIMER FOR LLM ARCHITECTURE")
    print("=" * 70)

    demo_tensor_creation_and_devices()
    demo_memory_strides_and_views()
    demo_broadcasting_mechanics()
    demo_autograd_engine()
    demo_custom_nn_module()

    print("")
    print("=" * 70)
    print("💡 KEY PEDAGOGICAL TAKEAWAYS:")
    print("=" * 70)
    print("1. Tensors are flat memory storage arrays viewed through shape and stride metadata.")
    print("2. Transpositions and views are zero-copy; use .contiguous() when reshaping.")
    print("3. Broadcasting automatically expands singleton dimensions without memory duplication.")
    print("4. Autograd dynamically constructs computation graphs for automatic differentiation.")
    print("5. Use torch.no_grad() during inference to disable graph building and maximize speed.")
    print("👉 Next: Module 00 introduces the foundations of language modeling!")
    print("=" * 70)


if __name__ == "__main__":
    main()
