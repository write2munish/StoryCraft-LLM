"""
Download a subset of the TinyStories dataset from Hugging Face for StoryCraft-LLM training.
"""

import argparse
import os
from datasets import load_dataset
from tqdm import tqdm

def download_data(num_samples: int = 100000, output_dir: str = "./raw"):
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "tinystories_train.txt")
    valid_path = os.path.join(output_dir, "tinystories_valid.txt")

    print(f"Loading TinyStories stream from Hugging Face...")
    ds = load_dataset("roneneldan/TinyStories", streaming=True)

    print(f"Writing {num_samples} training stories to {train_path}...")
    with open(train_path, "w", encoding="utf-8") as f:
        count = 0
        for sample in tqdm(ds["train"], total=num_samples, desc="Downloading train"):
            story = sample["text"].strip()
            if story:
                f.write(story + "\n<|endofstory|>\n")
                count += 1
            if count >= num_samples:
                break

    valid_samples = max(2000, num_samples // 20)
    print(f"Writing {valid_samples} validation stories to {valid_path}...")
    with open(valid_path, "w", encoding="utf-8") as f:
        count = 0
        for sample in tqdm(ds["validation"], total=valid_samples, desc="Downloading valid"):
            story = sample["text"].strip()
            if story:
                f.write(story + "\n<|endofstory|>\n")
                count += 1
            if count >= valid_samples:
                break

    print(f"Done! Datasets saved successfully in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TinyStories dataset subset")
    parser.add_argument("--num_samples", type=int, default=100000, help="Number of training stories to download")
    parser.add_argument("--output_dir", type=str, default="./raw", help="Target directory for raw text files")
    args = parser.parse_args()
    download_data(args.num_samples, args.output_dir)
