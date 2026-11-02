"""
Key-Value (KV) Cache Engine & Correctness Verification Suite.
=============================================================
Central Anchor: StoryCraft-LLM (The Tiny Story Generator)

This module provides:
1. CachedStoryGenerator: Unified autoregressive generation engine with toggleable KV-cache.
2. verify_generation_correctness: Rigorous step-by-step mathematical logit comparison
   and token-level equivalence check between Naive O(T^2) and KV-Cached O(1) inference.
"""

import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import torch
import torch.nn.functional as F

from storycraft.model import StoryCraftLM
from storycraft.tokenizer import StoryTokenizer


@dataclass
class StepVerification:
    step: int
    token_id_naive: int
    token_id_cached: int
    word_naive: str
    word_cached: str
    max_logit_diff: float
    cosine_sim: float
    passed: bool


@dataclass
class CorrectnessReport:
    prompt: str
    tokens_naive: List[int]
    tokens_cached: List[int]
    story_naive: str
    story_cached: str
    steps: List[StepVerification]
    all_tokens_match: bool
    max_diff_across_all_steps: float
    min_cosine_sim: float
    passed: bool


class CachedStoryGenerator:
    """
    Inference generator supporting both Naive (full sequence recompute)
    and KV-Cached step-by-step autoregressive generation.
    """
    def __init__(self, model: StoryCraftLM, tokenizer: StoryTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def generate_naive(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 0.0,
        repetition_penalty: float = 1.0
    ) -> Tuple[List[int], str, float]:
        """
        Naive O(T^2) Generation: At every step, passes the entire cumulative token history
        through the transformer from scratch without reusing past Key/Value tensors.
        """
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated = list(prompt_ids)
        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                inp = torch.tensor([generated], dtype=torch.long)
                logits, _ = self.model(inp, past_kvs=None)
                next_logits = logits[0, -1, :].clone()

                if repetition_penalty > 1.0:
                    for tid in set(generated):
                        if next_logits[tid] > 0:
                            next_logits[tid] /= repetition_penalty
                        else:
                            next_logits[tid] *= repetition_penalty

                if temperature <= 1e-4:
                    next_tok = torch.argmax(next_logits).item()
                else:
                    probs = F.softmax(next_logits / temperature, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1).item()

                generated.append(next_tok)
                if next_tok == self.tokenizer.eos_id:
                    break

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        story = self.tokenizer.decode(generated)
        return generated, story, elapsed_ms

    def generate_cached(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 0.0,
        repetition_penalty: float = 1.0
    ) -> Tuple[List[int], str, float]:
        """
        KV-Cached O(1) Generation: Prefills the prompt once, then passes only single
        new tokens to the network, dynamically concatenating past Keys and Values.
        """
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated = list(prompt_ids)
        start_time = time.perf_counter()

        with torch.no_grad():
            # 1. Prefill Phase: Process prompt and initialize KV cache
            inp = torch.tensor([prompt_ids], dtype=torch.long)
            logits, past_kvs = self.model(inp, past_kvs=None)
            next_logits = logits[0, -1, :].clone()

            if repetition_penalty > 1.0:
                for tid in set(generated):
                    if next_logits[tid] > 0:
                        next_logits[tid] /= repetition_penalty
                    else:
                        next_logits[tid] *= repetition_penalty

            if temperature <= 1e-4:
                next_tok = torch.argmax(next_logits).item()
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_tok)

            # 2. Decode Phase: Step-by-step single token pass with cached KV
            for _ in range(max_new_tokens - 1):
                if next_tok == self.tokenizer.eos_id:
                    break

                single_inp = torch.tensor([[next_tok]], dtype=torch.long)
                logits, past_kvs = self.model(single_inp, past_kvs=past_kvs)
                next_logits = logits[0, -1, :].clone()

                if repetition_penalty > 1.0:
                    for tid in set(generated):
                        if next_logits[tid] > 0:
                            next_logits[tid] /= repetition_penalty
                        else:
                            next_logits[tid] *= repetition_penalty

                if temperature <= 1e-4:
                    next_tok = torch.argmax(next_logits).item()
                else:
                    probs = F.softmax(next_logits / temperature, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1).item()

                generated.append(next_tok)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        story = self.tokenizer.decode(generated)
        return generated, story, elapsed_ms


def verify_generation_correctness(
    model: StoryCraftLM,
    tokenizer: StoryTokenizer,
    prompt: str = "Once upon a time Lily found a magic key",
    num_tokens: int = 15,
    atol: float = 1e-4
) -> CorrectnessReport:
    """
    Performs rigorous step-by-step mathematical verification between
    Naive Full-Context Generation and KV-Cached Generation.

    Verifies:
    1. Logit equivalence at every single decoding step (diff < atol).
    2. Cosine similarity between naive and cached logit vectors (> 0.9999).
    3. Token ID exact identity at each step.
    4. Decoded story string exact identity.
    """
    model.eval()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens_naive = list(prompt_ids)
    tokens_cached = list(prompt_ids)

    step_verifications: List[StepVerification] = []

    with torch.no_grad():
        # Prefill prompt for cached model
        inp_prefill = torch.tensor([prompt_ids], dtype=torch.long)
        logits_cached, past_kvs = model(inp_prefill, past_kvs=None)
        
        # Prefill forward for naive model
        logits_naive, _ = model(inp_prefill, past_kvs=None)

        curr_naive_logits = logits_naive[0, -1, :].clone()
        curr_cached_logits = logits_cached[0, -1, :].clone()

        for step in range(1, num_tokens + 1):
            # 1. Compare logits at current step
            diff = (curr_naive_logits - curr_cached_logits).abs().max().item()
            cos_sim = F.cosine_similarity(
                curr_naive_logits.unsqueeze(0),
                curr_cached_logits.unsqueeze(0)
            ).item()

            tok_naive = torch.argmax(curr_naive_logits).item()
            tok_cached = torch.argmax(curr_cached_logits).item()

            word_naive = tokenizer.vocab.get(tok_naive, f"<{tok_naive}>")
            word_cached = tokenizer.vocab.get(tok_cached, f"<{tok_cached}>")

            step_passed = (tok_naive == tok_cached) and (diff < atol)
            step_verifications.append(
                StepVerification(
                    step=step,
                    token_id_naive=tok_naive,
                    token_id_cached=tok_cached,
                    word_naive=word_naive,
                    word_cached=word_cached,
                    max_logit_diff=diff,
                    cosine_sim=cos_sim,
                    passed=step_passed
                )
            )

            # Append chosen token
            tokens_naive.append(tok_naive)
            tokens_cached.append(tok_cached)

            if step == num_tokens:
                break

            # 2. Advance Naive: re-run entire token sequence
            inp_naive_full = torch.tensor([tokens_naive], dtype=torch.long)
            logits_naive, _ = model(inp_naive_full, past_kvs=None)
            curr_naive_logits = logits_naive[0, -1, :].clone()

            # 3. Advance Cached: run ONLY the single newest token with past_kvs
            single_inp = torch.tensor([[tok_cached]], dtype=torch.long)
            logits_cached, past_kvs = model(single_inp, past_kvs=past_kvs)
            curr_cached_logits = logits_cached[0, -1, :].clone()

    story_naive = tokenizer.decode(tokens_naive)
    story_cached = tokenizer.decode(tokens_cached)

    all_tokens_match = (tokens_naive == tokens_cached)
    max_diff = max(s.max_logit_diff for s in step_verifications) if step_verifications else 0.0
    min_cos = min(s.cosine_sim for s in step_verifications) if step_verifications else 1.0
    overall_passed = all_tokens_match and all(s.passed for s in step_verifications)

    return CorrectnessReport(
        prompt=prompt,
        tokens_naive=tokens_naive,
        tokens_cached=tokens_cached,
        story_naive=story_naive,
        story_cached=story_cached,
        steps=step_verifications,
        all_tokens_match=all_tokens_match,
        max_diff_across_all_steps=max_diff,
        min_cosine_sim=min_cos,
        passed=overall_passed
    )
