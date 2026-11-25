"""
EE471 Week 9 — Task 1.3: Text Generation
Completes the given prompt with max length 35 words and returns 2 alternatives.
"""

from transformers import pipeline, set_seed


def main():
    set_seed(42)
    generator = pipeline("text-generation", model="gpt2")

    prompt = "If I continue to complete all in-class exercises in this modern software practices course,"
    outputs = generator(
        prompt,
        max_new_tokens=35,
        num_return_sequences=2,
        do_sample=True,
        pad_token_id=50256,
    )

    for idx, out in enumerate(outputs, 1):
        words = out["generated_text"].split()[:35]
        print(f"--- Alternative {idx} ---")
        print(" ".join(words) + "\n")


if __name__ == "__main__":
    main()
