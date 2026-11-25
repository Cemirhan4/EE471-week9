"""
EE471 Week 9 — Task 1.4: Mask Filling (Fill-Mask)
Fills the gap in: "To understand generative AI, one must study […] well."
"""

from transformers import pipeline


def main():
    unmasker = pipeline("fill-mask", model="distilroberta-base")
    mask_token = unmasker.tokenizer.mask_token

    sentence = f"To understand generative AI, one must study {mask_token} well."
    results = unmasker(sentence, top_k=3)

    print(f"Prompt: {sentence}\n")
    for res in results:
        print(f"Token: {res['token_str']:15s} | Score: {res['score']:.4f} | Sequence: {res['sequence']}")


if __name__ == "__main__":
    main()
