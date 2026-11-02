"""
Module 06: Discriminative LLMs - Story Classification & Moderation
===================================================================
Central Anchor: StoryCraft-LLM (The Tiny Story Generator)

This script demonstrates:
1. Loading the persistent pre-trained backbone: checkpoints/storycraft_base.pt.
2. Converting a causal autoregressive LM into a sequence classifier (The Dual LLM Paradigm).
3. Extracting the last token representation h[:, -1, :] to aggregate full narrative context.
4. Multi-class story genre classification and safety content moderation.
5. Saving the fine-tuned classifier to checkpoints/storycraft_classifier.pt.

Run this script:
    python3 modules/06_classification/demo_story_classifier.py
"""

import os
import sys
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)
from storycraft import StoryTokenizer, StoryCraftLM


class StoryClassifier(nn.Module):
    def __init__(self, base_model: StoryCraftLM, num_genres: int = 4, num_safety: int = 2):
        super().__init__()
        self.base_model = base_model
        d_model = base_model.d_model
        self.genre_head = nn.Linear(d_model, num_genres)
        self.safety_head = nn.Linear(d_model, num_safety)

    def freeze_backbone(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        print("   [Classifier] Base backbone weights frozen. Only classification heads will update.")

    def forward(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.base_model.tok_embed(token_ids)
        for block in self.base_model.blocks:
            h, _ = block(h)
        h = self.base_model.norm_final(h)
        last_tok = h[:, -1, :]  # Causal representation of full story
        return {
            "genre_logits": self.genre_head(last_tok),
            "safety_logits": self.safety_head(last_tok)
        }


def main():
    print("=" * 70)
    print("🏷️ MODULE 06: DISCRIMINATIVE LLMs - STORY CLASSIFIER & MODERATION")
    print("=" * 70)

    chk_dir = os.path.join(repo_root, "checkpoints")
    tok_path = os.path.join(chk_dir, "tokenizer.pkl")
    base_path = os.path.join(chk_dir, "storycraft_base.pt")
    out_clf_path = os.path.join(chk_dir, "storycraft_classifier.pt")

    genre_labels = ["Adventure", "Bedtime", "Animal", "Fantasy"]
    safety_labels = ["Safe", "Unsafe / Scary"]

    # 1. Load Pretrained Backbone from Module 04
    print("\n1. Loading Pretrained Backbone Checkpoint:")
    if os.path.exists(tok_path) and os.path.exists(base_path):
        tokenizer = StoryTokenizer.from_pretrained(tok_path)
        ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
        base_model = StoryCraftLM(**ckpt["config"])
        base_model.load_state_dict(ckpt["model_state_dict"])
        print(f"   ✓ Successfully loaded backbone from: {base_path}")
    else:
        tokenizer = StoryTokenizer(target_vocab_size=128)
        base_model = StoryCraftLM(vocab_size=128, d_model=64, n_layers=2, n_heads=2)

    model = StoryClassifier(base_model, num_genres=len(genre_labels), num_safety=len(safety_labels))
    model.freeze_backbone()

    # 2. Labeled Training Samples
    sample_stories = [
        ("Once upon a time a knight climbed a rocky mountain to find treasure.", 0, 0),
        ("A sleepy little bear yawned under the warm starry night sky.", 1, 0),
        ("A puppy chased a butterfly across the sunny grassy garden meadow.", 2, 0),
        ("A scary monster with sharp fangs jumped out to attack the village.", 3, 1),
    ]

    # Tokenize samples
    batch_tokens = []
    max_len = 16
    for text, _, _ in sample_stories:
        ids = tokenizer.encode(text, add_special_tokens=False)[:max_len]
        while len(ids) < max_len:
            ids.append(tokenizer.pad_id)
        batch_tokens.append(ids)

    x = torch.tensor(batch_tokens, dtype=torch.long)
    target_genres = torch.tensor([s[1] for s in sample_stories], dtype=torch.long)
    target_safety = torch.tensor([s[2] for s in sample_stories], dtype=torch.long)

    # 3. Fine-tuning Classification Heads
    print("\n2. Fine-Tuning Classification Heads on Story Dataset:")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, 4):
        optimizer.zero_grad()
        out = model(x)
        l_genre = loss_fn(out["genre_logits"], target_genres)
        l_safety = loss_fn(out["safety_logits"], target_safety)
        total_loss = l_genre + l_safety
        total_loss.backward()
        optimizer.step()
        print(f"   Epoch {epoch}: Total Loss = {total_loss.item():.4f} (Genre: {l_genre.item():.4f}, Safety: {l_safety.item():.4f})")

    # 4. Save Classifier Checkpoint
    torch.save({
        "classifier_state_dict": model.state_dict(),
        "genre_labels": genre_labels,
        "safety_labels": safety_labels
    }, out_clf_path)
    print(f"\n3. Persistent Checkpoint Saved:")
    print(f"   ✓ Saved classifier model to: {out_clf_path}")

    # 5. Predictions on Test Story
    print("\n4. Testing Classification & Moderation on a New Story:")
    test_story = "Lily and her puppy chased a red ball in the sunny garden."
    test_ids = tokenizer.encode(test_story, add_special_tokens=False)[:max_len]
    test_tensor = torch.tensor([test_ids], dtype=torch.long)

    with torch.no_grad():
        preds = model(test_tensor)
        pred_genre = genre_labels[torch.argmax(preds["genre_logits"]).item()]
        pred_safety = safety_labels[torch.argmax(preds["safety_logits"]).item()]

    print(f"   Story: \"{test_story}\"")
    print(f"   -> Predicted Genre:  {pred_genre}")
    print(f"   -> Content Safety:   {pred_safety}")

    print("\n" + "=" * 70)
    print("💡 KEY PEDAGOGICAL TAKEAWAYS:")
    print("=" * 70)
    print("1. Module 06 loads the exact checkpoints/storycraft_base.pt saved by Module 04.")
    print("2. The frozen backbone provides rich representations while lightweight heads train fast.")
    print("3. Persists checkpoints/storycraft_classifier.pt for downstream automated filtering!")
    print("=" * 70)


if __name__ == "__main__":
    main()
