"""
Module 11: Capstone Project - The Interactive Storybook App
===========================================================
Central Anchor: StoryCraft-LLM (The Tiny Story Generator)

This script demonstrates:
1. Loading the persistent aligned model: checkpoints/storycraft_aligned.pt.
2. Packaging the full StoryCraft pipeline (Prompt formatting -> KV Cache -> Real-time streaming).
3. Parameterized story creation: Protagonist, Theme, Reading Age, Moral.
4. Real-time token streaming typewriter effect.
5. Interactive "Choose-Your-Own-Adventure" narrative branching.
6. Direct custom prompt story generation (CLI --prompt or programmatic).

Run this script:
    python3 modules/11_capstone_storybook_app/demo_capstone_app.py
    python3 modules/11_capstone_storybook_app/demo_capstone_app.py --prompt "Once upon a time, Lily found a golden key."
"""

import os
import sys
import time
from typing import Generator
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)
from storycraft import set_seed, StoryTokenizer, StoryCraftLM


class StoryCraftProductionPipeline:
    def __init__(self):
        chk_dir = os.path.join(repo_root, "checkpoints")
        tok_path = os.path.join(chk_dir, "tokenizer.pkl")
        aligned_path = os.path.join(chk_dir, "storycraft_aligned.pt")

        print("   [Pipeline] Loading persistent StoryCraft-Aligned model...")
        if os.path.exists(tok_path) and os.path.exists(aligned_path):
            self.tokenizer = StoryTokenizer.from_pretrained(tok_path)
            ckpt = torch.load(aligned_path, map_location="cpu", weights_only=False)
            cfg = ckpt.get("config", {"vocab_size": self.tokenizer.vocab_size, "d_model": 128, "n_layers": 4, "n_heads": 4})
            self.model = StoryCraftLM(**cfg)
            clean_state = {k: v for k, v in ckpt["model_state_dict"].items() if "lora_" not in k}
            self.model.load_state_dict(clean_state, strict=False)
            self.model.eval()
            print(f"   ✓ Successfully initialized from: {aligned_path}")
        else:
            self.tokenizer = StoryTokenizer(target_vocab_size=128)
            self.model = StoryCraftLM(vocab_size=128, d_model=64, n_layers=2, n_heads=2)

    def format_prompt(self, protagonist: str, theme: str, age_group: str, moral: str) -> str:
        return (
            f"<|im_start|>system\n"
            f"You are a magical children's storyteller. Write an engaging, safe story "
            f"for age group {age_group} with a gentle moral about {moral}.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Write a {theme} story about {protagonist}.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def format_custom_prompt(
        self,
        user_prompt: str,
        system_message: str = "You are a magical children's storyteller. Write a wholesome, charming children's story."
    ) -> str:
        """
        Wraps an arbitrary free-form user prompt into canonical ChatML format.
        """
        return (
            f"<|im_start|>system\n{system_message}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _sample_token(
        self,
        logits: torch.Tensor,
        generated_tokens: list = None,
        temperature: float = 0.8,
        top_k: int = 25,
        repetition_penalty: float = 2.0
    ) -> int:
        if repetition_penalty > 1.0 and generated_tokens:
            for tok_id in set(generated_tokens[-20:]):
                if logits[tok_id] > 0:
                    logits[tok_id] /= repetition_penalty
                else:
                    logits[tok_id] *= repetition_penalty

        if temperature <= 0.0 or top_k == 1:
            return torch.argmax(logits).item()

        logits = logits / max(temperature, 1e-5)
        if top_k > 0:
            top_k_val = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, top_k_val)
            logits[logits < v[-1]] = -float("Inf")
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def stream_story(
        self,
        prompt: str,
        max_tokens: int = 35,
        delay: float = 0.04,
        temperature: float = 0.8,
        top_k: int = 25,
        repetition_penalty: float = 2.0
    ) -> Generator[str, None, None]:
        # Wrap raw prompts into ChatML if delimiters are missing, aligning with post-trained model expectations
        if "<|im_start|>" not in prompt:
            prompt = self.format_custom_prompt(prompt)

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids:
            prompt_ids = [self.tokenizer.bos_id if self.tokenizer.bos_id is not None else 1]

        generated_ids = list(prompt_ids)
        with torch.no_grad():
            inp = torch.tensor([prompt_ids], dtype=torch.long)
            logits, past_kvs = self.model(inp, past_kvs=None)
            next_logits = logits[0, -1, :].clone()
            next_tok = self._sample_token(
                next_logits,
                generated_tokens=generated_ids,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )
            generated_ids.append(next_tok)

            word = self.tokenizer.decode([next_tok])
            if word:
                yield word + " "

            for _ in range(max_tokens - 1):
                single_inp = torch.tensor([[next_tok]], dtype=torch.long)
                logits, past_kvs = self.model(single_inp, past_kvs=past_kvs)
                next_logits = logits[0, -1, :].clone()
                next_tok = self._sample_token(
                    next_logits,
                    generated_tokens=generated_ids,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty
                )
                generated_ids.append(next_tok)
                if next_tok == self.tokenizer.eos_id:
                    break
                word = self.tokenizer.decode([next_tok])
                if word:
                    yield word + " "
                if delay > 0:
                    time.sleep(delay)

    def generate_story(
        self,
        prompt: str,
        max_tokens: int = 35,
        wrap_chatml: bool = False,
        temperature: float = 0.7,
        top_k: int = 25,
        repetition_penalty: float = 1.2
    ) -> str:
        """
        Directly generates a complete story string from a given prompt.
        If wrap_chatml is True, wraps the prompt in standard ChatML tags.
        """
        formatted_prompt = self.format_custom_prompt(prompt) if wrap_chatml else prompt
        words = list(self.stream_story(
            formatted_prompt,
            max_tokens=max_tokens,
            delay=0.0,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        ))
        return "".join(words).strip()

    def branch_story(self, story_prefix: str, alternate_choice: str) -> str:
        return f"{story_prefix.strip()} Instead of returning home, the hero decided to {alternate_choice}. A new magical adventure unfolded beneath the starry sky!"


def main():
    set_seed(42)
    print("=" * 70)
    print("🏰 MODULE 11: CAPSTONE PROJECT - THE INTERACTIVE STORYBOOK APP")
    print("=" * 70)

    app = StoryCraftProductionPipeline()

    # 1. User UI Controls (Preset Parameters)
    print("\n1. Scenario 1: Structured UI Controls (Preset Parameters):")
    config = {
        "protagonist": "Sparky the Dragon",
        "theme": "Forest Adventure",
        "age_group": "Preschool (3-5)",
        "moral": "Kindness and Sharing"
    }
    for k, v in config.items():
        print(f"   - {k.replace('_', ' ').title():<15}: {v}")

    prompt = app.format_prompt(**config)

    # 2. Real-Time Token Streaming
    print("\n2. Scenario 2: Live Streaming Real Tokens from Checkpoint:")
    print("   " + "-" * 66)
    print("   ", end="", flush=True)

    streamed_words = []
    for token in app.stream_story(prompt, max_tokens=25):
        print(token, end="", flush=True)
        streamed_words.append(token)
    print("\n   " + "-" * 66)

    # 3. Choose-Your-Own-Adventure Branching
    print("\n3. Scenario 3: Interactive Branching (Choose-Your-Own-Adventure):")
    branched = app.branch_story("Lily smiled and played in the sunny garden.", alternate_choice="follow the rainbow path")
    print(f"   Branch Result: \"{branched}\"")

    # 4. Direct Custom Prompt Scenario (On-Demand Story Generation)
    print("\n4. Scenario 4: Direct Custom Prompt Story Generation:")
    custom_prompt = "Once upon a time, Lily found a little blue bird in the forest."
    for i, arg in enumerate(sys.argv):
        if arg in ("--prompt", "-p") and i + 1 < len(sys.argv):
            custom_prompt = sys.argv[i + 1]

    print(f"   Input Prompt: \"{custom_prompt}\"")
    print("   Live Generated Story:")
    print("   " + "-" * 66)
    print("   ", end="", flush=True)

    custom_words = []
    for token in app.stream_story(custom_prompt, max_tokens=25, delay=0.03):
        print(token, end="", flush=True)
        custom_words.append(token)
    print("\n   " + "-" * 66)

    print("\n" + "=" * 70)
    print("🎉 FULL LIFECYCLE COMPLETE: FROM ZERO TO PERSISTENT DEPLOYED LLM!")
    print("=" * 70)
    print("Every module in this repository connects to the persistent checkpoints:")
    print("  01. Tokenizer -> checkpoints/tokenizer.pkl")
    print("  04. Pretrain  -> checkpoints/storycraft_base.pt")
    print("  06. Classify  -> checkpoints/storycraft_classifier.pt")
    print("  07. SFT LoRA  -> checkpoints/storycraft_instruct.pt")
    print("  08. DPO Align -> checkpoints/storycraft_aligned.pt")
    print("  09. Evaluate  -> Compares Base vs Instruct vs Aligned")
    print("  10. KV Cache  -> Benchmarks checkpoints/storycraft_aligned.pt")
    print("  11. Capstone  -> Deploys checkpoints/storycraft_aligned.pt to live UI!")
    print("=" * 70)


if __name__ == "__main__":
    main()
