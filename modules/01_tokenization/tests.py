import unittest
import sys
import os
import torch
sys.path.insert(0, os.path.dirname(__file__))
from demo_tokenization import MiniBPETokenizer, StoryCausalDataset

class TestTokenization(unittest.TestCase):
    def setUp(self):
        self.corpus = "Lily loved her red ball. One day Lily saw a red dog."
        self.tokenizer = MiniBPETokenizer(target_vocab_size=35)
        self.tokenizer.train(self.corpus)

    def test_special_tokens_exist(self):
        self.assertIn("<|pad|>", self.tokenizer.inv_vocab)
        self.assertIn("<|bos|>", self.tokenizer.inv_vocab)
        self.assertIn("<|eos|>", self.tokenizer.inv_vocab)

    def test_encoding_structure(self):
        tokens = self.tokenizer.encode("Lily loved red")
        # Starts with <|bos|> and ends with <|eos|>
        self.assertEqual(tokens[0], self.tokenizer.inv_vocab["<|bos|>"])
        self.assertEqual(tokens[-1], self.tokenizer.inv_vocab["<|eos|>"])
        self.assertGreater(len(tokens), 2)

    def test_roundtrip_decoding(self):
        tokens = self.tokenizer.encode("Lily red ball")
        decoded = self.tokenizer.decode(tokens)
        self.assertIn("lily", decoded.lower())
        self.assertIn("red", decoded.lower())

    def test_causal_target_shift(self):
        stream = [1, 10, 20, 30, 40, 50, 60]
        dataset = StoryCausalDataset(stream, context_len=4)
        x, y = dataset[0]
        # Target y must be exactly x shifted by 1
        self.assertTrue(torch.equal(x[1:], y[:-1]))
        self.assertEqual(y[-1].item(), stream[4])

if __name__ == "__main__":
    unittest.main()
