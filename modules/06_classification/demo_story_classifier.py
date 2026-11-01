"""
Module 06: Discriminative LLMs - Story Classification & Moderation
===================================================================
Central Anchor: StoryCraft-LLM (The Tiny Story Generator)

This script demonstrates:
1. Adapting a causal autoregressive LLM into a sequence classifier.
2. Why the last token representation h[:, -1, :] aggregates the whole story.
3. Multi-class story genre classification (Adventure, Bedtime, Animal, Fantasy).
4. Safety / Content Moderation filter (Safe vs Unsafe/Scary).
5. Backbone parameter freezing vs training the classification head.

Run this script:
    python3 modules/06_classification/demo_story_classifier.py
"""

from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Story Classification Architecture
# ============================================================================
class StoryClassifier(nn.Module):
    """
    Wraps a causal Transformer backbone for discriminative sequence classification.
    Takes story tokens, runs through Transformer blocks, extracts the last token vector,
    and projects into K genre classes and 2 safety classes.
    """
    def __init__(self, vocab_size: int = 1000, d_model: int = 64, num_genres: int = 4):
        super().__init__()
        # Backbone (Simulated minimal causal transformer)
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True
        )

        # Classification Heads attached to the last token vector
        self.genre_head = nn.Linear(d_model, num_genres)      # Multi-class
        self.safety_head = nn.Linear(d_model, 2)              # Binary (Safe / Unsafe)

    def freeze_backbone(self):
        """Freeze embedding and transformer layers for head-only fine-tuning."""
        for param in self.tok_embed.parameters():
            param.requires_grad = False
        for param in self.encoder_layer.parameters():
            param.requires_grad = False
        print("[Classifier] Backbone weights frozen. Only classification heads will train.")

    def forward(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        # token_ids: [Batch, Sequence_Len]
        hidden_states = self.tok_embed(token_ids)
        hidden_states = self.encoder_layer(hidden_states)  # [B, T, d_model]

        # Extract the representation of the LAST token: [B, d_model]
        # In a causal decoder, the last token has attended to all prior tokens!
        last_token_repr = hidden_states[:, -1, :]

        genre_logits = self.genre_head(last_token_repr)    # [B, num_genres]
        safety_logits = self.safety_head(last_token_repr)  # [B, 2]
        return {
            "genre_logits": genre_logits,
            "safety_logits": safety_logits
        }


# ============================================================================
# 2. Demonstration Runner
# ============================================================================
def main():
    print("=" * 70)
    print("🏷️ MODULE 06: DISCRIMINATIVE LLMs - STORY CLASSIFIER & MODERATION")
    print("=" * 70)

    genre_labels = ["Adventure", "Bedtime", "Animal", "Fantasy"]
    safety_labels = ["Safe", "Unsafe / Scary"]

    model = StoryClassifier(vocab_size=1000, d_model=64, num_genres=len(genre_labels))
    
    # Check parameter counts
    backbone_params = sum(p.numel() for p in model.tok_embed.parameters()) + sum(p.numel() for p in model.encoder_layer.parameters())
    head_params = sum(p.numel() for p in model.genre_head.parameters()) + sum(p.numel() for p in model.safety_head.parameters())

    print("\n1. Model Architecture & Dual-Use Transition:")
    print(f"   - Backbone parameters (Language Model): {backbone_params:,}")
    print(f"   - Classification Head parameters:       {head_params:,}")
    print("   Notice that the classification heads require less than 1% of the model weights!")

    # 2. Freeze backbone
    print("\n2. Freezing Backbone for Efficient Fine-Tuning:")
    model.freeze_backbone()

    # 3. Simulated Stories and Labels
    print("\n3. Classifying Incoming Story Samples:")
    sample_stories = [
        ("Once upon a time a knight climbed a steep rocky mountain to find treasure.", 0, 0),  # Adventure, Safe
        ("A sleepy little bear yawned and closed his eyes under the warm starry sky.", 1, 0),   # Bedtime, Safe
        ("A puppy and a kitten chased a butterfly across the sunny grassy meadow.", 2, 0),     # Animal, Safe
        ("A dark monster with sharp fangs jumped out of the shadow to destroy the village.", 3, 1), # Fantasy, Unsafe
    ]

    torch.manual_seed(42)
    # Simulate tokenized stories: Batch=4, Seq_len=12
    batch_input = torch.randint(10, 1000, (4, 12))
    genre_targets = torch.tensor([s[1] for s in sample_stories], dtype=torch.long)
    safety_targets = torch.tensor([s[2] for s in sample_stories], dtype=torch.long)

    # 4. Forward Pass & Loss
    outputs = model(batch_input)
    genre_probs = F.softmax(outputs["genre_logits"], dim=-1)
    safety_probs = F.softmax(outputs["safety_logits"], dim=-1)

    print("\n4. Predictions on Story Batch (Untrained Head):")
    for i, (text, true_genre, true_safety) in enumerate(sample_stories):
        pred_genre = torch.argmax(genre_probs[i]).item()
        pred_safety = torch.argmax(safety_probs[i]).item()
        print(f"\n   Story {i+1}: \"{text[:55]}...\"")
        print(f"   - Target: Genre='{genre_labels[true_genre]}', Safety='{safety_labels[true_safety]}'")
        print(f"   - Output: Genre='{genre_labels[pred_genre]}' (conf: {genre_probs[i, pred_genre]:.1%}), "
              f"Safety='{safety_labels[pred_safety]}' (conf: {safety_probs[i, pred_safety]:.1%})")

    # 5. One Step of Training on Classification Heads
    print("\n5. Running One Training Step (Updating Heads Only):")
    loss_fn = nn.CrossEntropyLoss()
    loss_genre = loss_fn(outputs["genre_logits"], genre_targets)
    loss_safety = loss_fn(outputs["safety_logits"], safety_targets)
    total_loss = loss_genre + loss_safety

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(f"   Combined Loss: {total_loss.item():.4f} (Genre: {loss_genre.item():.4f}, Safety: {loss_safety.item():.4f})")
    print("   Gradients computed ONLY on heads; backbone remained intact!")

    print("\n" + "=" * 70)
    print("💡 KEY PEDAGOGICAL TAKEAWAYS:")
    print("=" * 70)
    print("1. LLMs are not just text generators—they are general feature extractors.")
    print("2. The final token's hidden state aggregates the full narrative context.")
    print("3. Pre-trained weights can be repurposed for content moderation and metadata tagging")
    print("   with minimal training overhead.")
    print("👉 Next: Module 07 explores Instruction Tuning (SFT) and LoRA from scratch!")
    print("=" * 70)


if __name__ == "__main__":
    main()
