"""
EE471 Week 9 — Task 4.2: Text Encoding and Decoding
Demonstrates tokenizing text into tokens & input IDs, then decoding back to string.
"""

from transformers import AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    text = "Using a Transformer network is simple!"

    # Step 1: Tokenize into subword tokens
    tokens = tokenizer.tokenize(text)
    print("Original Text :", text)
    print("Subword Tokens:", tokens)

    # Step 2: Convert tokens to numerical IDs
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print("Token IDs     :", ids)

    # Step 3: Full encode (includes special tokens [CLS] and [SEP])
    encoded = tokenizer.encode(text)
    print("Full Encoded  :", encoded)

    # Step 4: Decode back to text
    decoded_full = tokenizer.decode(encoded)
    decoded_clean = tokenizer.decode(encoded, skip_special_tokens=True)
    print("Decoded (raw) :", decoded_full)
    print("Decoded (clean):", decoded_clean)


if __name__ == "__main__":
    main()
