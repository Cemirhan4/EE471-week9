"""
EE471 Week 9 — Task 1.8: Translation (French to English)
Translates "Ce cours est produit par Hugging Face." using AutoModelForSeq2SeqLM.
(Compatible with both Transformers v4 and v5)
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def main():
    model_name = "Helsinki-NLP/opus-mt-fr-en"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    sentence = "Ce cours est produit par Hugging Face."
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        translated_ids = model.generate(**inputs)

    english_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    print(f"French : {sentence}")
    print(f"English: {english_text}")


if __name__ == "__main__":
    main()
