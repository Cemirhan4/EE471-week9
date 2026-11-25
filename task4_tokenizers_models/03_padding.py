"""
EE471 Week 9 — Task 4.3: Input Padding
Demonstrates padding variable-length sequences into uniform tensors with attention masks.
"""

from transformers import AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    sentences = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]

    # Longest-sequence padding in batch
    batch_longest = tokenizer(sentences, padding=True, return_tensors="pt")
    print("=== Padding = True (pad to longest in batch) ===")
    print("Input IDs shape     :", batch_longest["input_ids"].shape)
    print("Input IDs:\n", batch_longest["input_ids"])
    print("Attention Mask:\n", batch_longest["attention_mask"])

    # Fixed max_length padding
    batch_fixed = tokenizer(sentences, padding="max_length", max_length=20, return_tensors="pt")
    print("\n=== Padding = 'max_length' (max_length=20) ===")
    print("Input IDs shape     :", batch_fixed["input_ids"].shape)
    print("Attention Mask:\n", batch_fixed["attention_mask"])


if __name__ == "__main__":
    main()
