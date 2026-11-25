"""
EE471 Week 9 — Task 4.4: Input Truncation
Demonstrates truncating long sequences to a specified max_length.
"""

from transformers import AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    long_sentence = (
        "The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st century, "
        "often compared to the Great Depression of the 1930s, triggered by the US housing bubble."
    )

    full_ids = tokenizer.encode(long_sentence)
    print(f"Original token count: {len(full_ids)}")

    truncated = tokenizer(long_sentence, truncation=True, max_length=12, return_tensors="pt")
    truncated_ids = truncated["input_ids"][0].tolist()

    print(f"Truncated token count (max_length=12): {len(truncated_ids)}")
    print("Truncated IDs    :", truncated_ids)
    print("Decoded Truncated:", tokenizer.decode(truncated_ids))


if __name__ == "__main__":
    main()
