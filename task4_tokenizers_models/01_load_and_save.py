"""
EE471 Week 9 — Task 4.1: Load and Save a Model & Tokenizer
Demonstrates loading a pretrained model/tokenizer from HuggingFace Hub,
saving locally with save_pretrained(), and reloading from disk.
"""

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification


SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")


def main():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    print(f"1. Loading pretrained model & tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    print(f"2. Saving model and tokenizer to: {SAVE_DIR}")
    tokenizer.save_pretrained(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)
    print("   Saved files:", os.listdir(SAVE_DIR))

    print(f"3. Reloading model and tokenizer from local directory...")
    reloaded_tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
    reloaded_model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
    print(f"   Reloaded successfully! Model class: {type(reloaded_model).__name__}")
    print(f"   Vocab size: {reloaded_tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
