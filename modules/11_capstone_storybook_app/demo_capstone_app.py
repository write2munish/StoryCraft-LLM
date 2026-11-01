"""
Module 11: Capstone Project - The Interactive Storybook App
===========================================================
Central Anchor: StoryCraft-LLM (The Tiny Story Generator)

This script demonstrates:
1. Packaging StoryCraft-LLM into a full pipeline (Prompt conditioning -> Sampling -> Streaming).
2. Parameterized story creation: Protagonist, Theme, Reading Age, Moral lesson.
3. Simulated Server-Sent Events (SSE) word-by-word streaming generator.
4. "Choose-Your-Own-Adventure" interactive narrative branching.

Run this script:
    python3 modules/11_capstone_storybook_app/demo_capstone_app.py
"""

import time
from typing import Generator, List, Dict


# ============================================================================
# 1. StoryCraft Engine Pipeline
# ============================================================================
class StoryCraftPipeline:
    """
    High-level orchestration engine that converts user UI controls into
    structured prompts, runs inference, and streams story tokens in real-time.
    """
    def __init__(self):
        print("[StoryCraft Engine] Initializing StoryCraft-LLM (8.5M Aligned Checkpoint)...")

    def format_chatml_prompt(self, protagonist: str, theme: str, age_group: str, moral: str) -> str:
        """Construct structured ChatML instruction."""
        prompt = (
            f"<|im_start|>system\n"
            f"You are a magical children's storyteller. Write an engaging, safe story "
            f"suitable for age group {age_group} with a gentle moral about {moral}.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Write a {theme} story about a protagonist named {protagonist}.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    def stream_story(self, prompt: str) -> Generator[str, None, None]:
        """
        Simulates real-time token streaming from the KV-cached LLM with typewriter timing.
        """
        simulated_story = (
            "Once upon a time in a sunny green meadow, Lily the curious hedgehog was searching "
            "for shiny pebbles. Beneath a tall oak tree, she discovered a small squirrel named Pip "
            "who was crying softly because he dropped his acorns. Lily smiled warmly, shared her "
            "red berry snack, and helped Pip gather all his acorns back into his basket. "
            "Pip hugged Lily happily, and from that day forward, the two were the best of forest friends. "
            "Lily learned that a small act of kindness makes everyone's world brighter."
        )

        words = simulated_story.split()
        for word in words:
            yield word + " "
            time.sleep(0.04)  # Real-time streaming delay (simulate ~25 tokens/sec)

    def branch_story(self, original_story: str, branch_point_index: int, new_choice: str) -> str:
        """
        Allows the user to branch an existing story into an alternate storyline!
        """
        sentences = original_story.split(". ")
        prefix = ". ".join(sentences[:branch_point_index]) + "."
        
        alternate_continuation = (
            f" Instead of going home, Lily decided to {new_choice}. "
            "She followed a winding path of silver mushrooms and discovered a secret singing waterfall!"
        )
        return prefix + alternate_continuation


# ============================================================================
# 2. Demonstration Runner
# ============================================================================
def main():
    print("=" * 70)
    print("🏰 MODULE 11: CAPSTONE PROJECT - THE INTERACTIVE STORYBOOK APP")
    print("=" * 70)

    app = StoryCraftPipeline()

    # 1. User UI Input Controls
    print("\n1. Receiving User Storybook Controls:")
    user_inputs = {
        "protagonist": "Lily the Hedgehog",
        "theme": "Forest Adventure",
        "age_group": "Preschool (3-5 years)",
        "moral": "Kindness and Helping Others"
    }
    for k, v in user_inputs.items():
        print(f"   - {k.replace('_', ' ').title():<15}: {v}")

    # 2. Format ChatML
    prompt = app.format_chatml_prompt(**user_inputs)
    print("\n2. Formatted Model Context:")
    for line in prompt.strip().split("\n"):
        print(f"   {line}")

    # 3. Real-Time Token Streaming Simulation
    print("\n3. Streaming Generated Story (Real-time Typewriter Effect):")
    print("   " + "-" * 66)
    print("   ", end="", flush=True)

    collected_story = []
    for token in app.stream_story(prompt):
        print(token, end="", flush=True)
        collected_story.append(token)
    print("\n   " + "-" * 66)

    full_story = "".join(collected_story).strip()

    # 4. Interactive Branching (Choose-Your-Own-Adventure)
    print("\n4. Interactive Branching Feature (Choose-Your-Own-Adventure):")
    print("   User clicks sentence 2 and chooses: 'Explore the hidden mushroom cave'")
    branched_story = app.branch_story(full_story, branch_point_index=2, new_choice="explore the hidden mushroom cave")
    
    print("\n   --- Alternate Story Branch ---")
    print(f"   \"{branched_story}\"")

    print("\n" + "=" * 70)
    print("🎉 CONGRATULATIONS! YOU HAVE COMPLETED THE ENTIRE CURRICULUM:")
    print("=" * 70)
    print("You started with raw text and statistical Bigrams in Module 00.")
    print("You built a custom BPE tokenizer, continuous RoPE embeddings, and a")
    print("modern LLaMA-style Decoder Transformer from scratch.")
    print("You pre-trained on 100k stories, added classification heads, fine-tuned with LoRA,")
    print("aligned morals with DPO, benchmarked quality, added KV Caching,")
    print("and shipped a full-stack real-time AI Storytelling Application!")
    print("=" * 70)


if __name__ == "__main__":
    main()
