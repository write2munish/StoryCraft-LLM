"""
Module 00: Foundations of Language Modeling
=============================================
Central Anchor: StoryCraft-LLM (The Tiny Story Generator)

This script demonstrates:
1. The mathematical formulation of autoregressive language modeling:
   P(w_1, ..., w_T) = prod_{t=1}^T P(w_t | w_1, ..., w_{t-1})
2. A pure Python Bigram and Trigram statistical language model.
3. Why count-based models fail (Curse of Dimensionality, repetition loops,
   and lack of long-range semantic memory).

Run this script:
    python3 modules/00_foundations/demo_foundations.py
"""

import random
from collections import defaultdict, Counter
from typing import List, Dict, Tuple


# ============================================================================
# 1. Toy Story Corpus (Children's Stories)
# ============================================================================
TOY_STORIES = [
    "once upon a time there was a little girl named lily . lily loved to play with her red ball in the green garden .",
    "once upon a time a small dog named barnaby saw a shiny bird . the dog barked happily and ran around .",
    "once upon a time there was a friendly dragon who loved baking sweet cookies for all his forest friends .",
    "lily and the small dog went to the big park . they saw a blue pond and jumped with joy .",
    "one sunny afternoon barnaby the dog found a red ball in the tall green grass . he brought it to lily .",
    "the friendly dragon shared his warm cookies with lily and barnaby . everyone smiled and felt very happy ."
]


# ============================================================================
# 2. Bigram Story Generator
# ============================================================================
class BigramStoryModel:
    """
    A statistical Bigram language model.
    Models P(word_t | word_{t-1}) using frequency counts from the training corpus.
    """
    def __init__(self):
        # transitions[w_{t-1}][w_t] = count
        self.transitions: Dict[str, Counter] = defaultdict(Counter)
        self.vocab = set()

    def train(self, corpus: List[str]):
        """
        Count occurrences of word pairs (w_{t-1}, w_t).
        """
        print("[Bigram] Training on story corpus...")
        for story in corpus:
            tokens = story.split()
            self.vocab.update(tokens)
            for i in range(len(tokens) - 1):
                prev_word = tokens[i]
                next_word = tokens[i + 1]
                self.transitions[prev_word][next_word] += 1
        print(f"[Bigram] Discovered vocabulary size: {len(self.vocab)} unique words")

    def get_probabilities(self, prev_word: str) -> Dict[str, float]:
        """
        Convert raw counts to conditional probabilities:
        P(w_t | w_{t-1}) = Count(w_{t-1}, w_t) / Count(w_{t-1})
        """
        next_counts = self.transitions.get(prev_word, Counter())
        total = sum(next_counts.values())
        if total == 0:
            return {}
        return {word: count / total for word, count in next_counts.items()}

    def generate(self, prompt: str, max_words: int = 25) -> str:
        """
        Autoregressively sample next words until max_words or dead end.
        """
        current_word = prompt.split()[-1].lower()
        output = prompt.split()

        for _ in range(max_words):
            probs = self.get_probabilities(current_word)
            if not probs:
                break  # Dead end (unseen state)
            
            # Sample next word according to probability distribution
            words = list(probs.keys())
            weights = list(probs.values())
            next_word = random.choices(words, weights=weights, k=1)[0]
            
            output.append(next_word)
            current_word = next_word
            if next_word == ".":
                break

        return " ".join(output)


# ============================================================================
# 3. Trigram Story Generator
# ============================================================================
class TrigramStoryModel:
    """
    A statistical Trigram language model.
    Models P(w_t | w_{t-2}, w_{t-1}) by conditioning on a 2-word context window.
    """
    def __init__(self):
        # transitions[(w_{t-2}, w_{t-1})][w_t] = count
        self.transitions: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

    def train(self, corpus: List[str]):
        """Count occurrences of word triplets."""
        for story in corpus:
            tokens = story.split()
            for i in range(len(tokens) - 2):
                ctx = (tokens[i], tokens[i + 1])
                nxt = tokens[i + 2]
                self.transitions[ctx][nxt] += 1

    def generate(self, prompt: str, max_words: int = 25) -> str:
        words = prompt.split()
        if len(words) < 2:
            raise ValueError("Trigram prompt needs at least 2 words (e.g. 'once upon')")

        output = list(words)
        for _ in range(max_words):
            ctx = (output[-2].lower(), output[-1].lower())
            next_counts = self.transitions.get(ctx, Counter())
            if not next_counts:
                # OOV context failure: The Curse of Dimensionality!
                break
            
            candidates = list(next_counts.keys())
            weights = list(next_counts.values())
            nxt = random.choices(candidates, weights=weights, k=1)[0]
            output.append(nxt)
            if nxt == ".":
                break

        return " ".join(output)


# ============================================================================
# 4. Educational Demonstration Runner
# ============================================================================
def main():
    print("=" * 70)
    print("📖 MODULE 00: FOUNDATIONS OF LANGUAGE MODELING")
    print("=" * 70)
    print("\n1. Concept: Language Modeling as Next-Token Prediction")
    print("   Every language model decomposes text into sequential conditional probabilities:")
    print("   P('Once', 'upon', 'a', 'time') = P('Once') * P('upon'|'Once') * P('a'|'Once upon') * ...\n")

    # Train Bigram Model
    bigram_lm = BigramStoryModel()
    bigram_lm.train(TOY_STORIES)

    print("\n2. Inspecting Bigram Conditional Probabilities:")
    for test_word in ["once", "lily", "red"]:
        probs = bigram_lm.get_probabilities(test_word)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        readable = ", ".join([f"'{w}': {p:.2f}" for w, p in sorted_probs[:3]])
        print(f"   P(next_word | '{test_word}') => [{readable}]")

    print("\n3. Generating Stories with Bigram Model:")
    for i in range(3):
        story = bigram_lm.generate("once upon", max_words=20)
        print(f"   Sample {i+1}: \"{story}\"")

    # Train Trigram Model
    trigram_lm = TrigramStoryModel()
    trigram_lm.train(TOY_STORIES)
    print("\n4. Generating with Trigram Model (2-word context):")
    for i in range(2):
        story = trigram_lm.generate("once upon", max_words=20)
        print(f"   Trigram {i+1}: \"{story}\"")

    print("\n" + "=" * 70)
    print("⚠️ KEY PEDAGOGICAL TAKEAWAYS (Why N-Grams Fail):")
    print("=" * 70)
    print("1. THE CURSE OF DIMENSIONALITY:")
    print("   To model 10 words of context with a 10,000 word vocabulary requires")
    print("   10,000^10 possible transitions — more than the atoms in the universe!")
    print("2. ZERO GENERALIZATION:")
    print("   If the bigram model saw 'red ball', it learns NOTHING about 'blue ball'")
    print("   because words are treated as atomic, isolated strings with no vector meaning.")
    print("3. MEMORY COLLAPSE:")
    print("   The model cannot remember the story's main character across a sentence.")
    print("\n👉 SOLUTION: In Module 01, we convert words to Subword Tokens, and in")
    print("   Module 02, we map tokens to continuous Vector Spaces (Embeddings)!")
    print("=" * 70)


if __name__ == "__main__":
    main()
