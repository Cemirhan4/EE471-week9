"""
EE471 Week 9 — Task 4.5: Adding Special Tokens
Demonstrates inspecting default special tokens, adding new custom tokens,
and resizing model token embeddings to match the expanded vocabulary.
"""

from transformers import AutoTokenizer, AutoModel


def main():
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    print("Default special tokens:", tokenizer.all_special_tokens)
    print("Initial vocab size    :", len(tokenizer))

    # Add custom domain tokens and a new special token
    new_tokens = ["[IYTE]", "[EE471]"]
    num_added = tokenizer.add_tokens(new_tokens)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<COURSE_TAG>"]})

    # Resize model embedding matrix to match the updated tokenizer size
    model.resize_token_embeddings(len(tokenizer))
    print(f"Added {num_added} regular tokens + 1 special token.")
    print("Updated vocab size    :", len(tokenizer))

    sample = "<COURSE_TAG> Welcome to [EE471] at [IYTE]!"
    encoded = tokenizer.encode(sample)
    print("Sample :", sample)
    print("Encoded:", encoded)
    print("Tokens :", tokenizer.convert_ids_to_tokens(encoded))


if __name__ == "__main__":
    main()
