"""
StoryGenerator: Real autoregressive decoding engine for StoryCraftLM.
Implements Greedy Search, Temperature scaling, Top-k, Top-p, and Repetition Penalty.
"""

from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F


def apply_repetition_penalty(logits: torch.Tensor, generated_tokens: List[int], penalty: float = 1.2) -> torch.Tensor:
    """Penalizes logits of tokens that have already appeared in the output."""
    for token_id in set(generated_tokens):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9
) -> int:
    """Applies Temperature, Top-k, and Top-p (Nucleus) filtering and samples 1 token."""
    logits = logits / max(temperature, 1e-5)

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k)[0][-1]
        logits[logits < threshold] = float("-inf")

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


class StoryGenerator:
    """
    Inference and sampling engine wrapped around StoryCraftLM and StoryTokenizer.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def get_next_token_distribution(self, prompt: str, temperature: float = 1.0) -> Dict[str, float]:
        """
        Runs a forward pass on prompt and returns {word: probability} for top candidates.
        """
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        inp = torch.tensor([token_ids], dtype=torch.long)
        with torch.no_grad():
            logits, _ = self.model(inp)
            next_logits = logits[0, -1, :].clone() / max(temperature, 1e-5)
            probs = F.softmax(next_logits, dim=-1)

        top_probs, top_indices = torch.topk(probs, k=min(10, self.tokenizer.vocab_size))
        result = {}
        for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
            word = self.tokenizer.vocab.get(idx, f"<token_{idx}>")
            result[word] = p
        return result

    def apply_repetition_penalty(self, logits: torch.Tensor, generated_tokens: List[int], penalty: float = 1.2) -> torch.Tensor:
        return apply_repetition_penalty(logits, generated_tokens, penalty)

    def sample_step(self, logits: torch.Tensor, temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9) -> int:
        return sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        strategy: str = "sample"
    ) -> str:
        """
        Autoregressively generates story continuation.
        strategy: 'greedy' or 'sample'.
        """
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated = list(prompt_ids)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                inp = torch.tensor([generated], dtype=torch.long)
                logits, _ = self.model(inp)
                next_logits = logits[0, -1, :].clone()

                if strategy == "greedy":
                    next_id = torch.argmax(next_logits).item()
                else:
                    if repetition_penalty > 1.0:
                        next_logits = self.apply_repetition_penalty(next_logits, generated, repetition_penalty)
                    next_id = self.sample_step(next_logits, temperature=temperature, top_k=top_k, top_p=top_p)

                generated.append(next_id)
                if next_id == self.tokenizer.eos_id:
                    break

        return self.tokenizer.decode(generated)
