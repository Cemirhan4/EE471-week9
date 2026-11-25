"""
EE471 Week 9 — Task 1.7: Text Summarization
Summarizes the 2008 Global Financial Crisis paragraph using AutoModelForSeq2SeqLM.
(Compatible with both Transformers v4 and v5)
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def main():
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    text = (
        "The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st century, "
        "often compared to the Great Depression of the 1930s. Triggered by the bursting of the United States "
        "housing bubble, its effects rippled across the globe, leading to the collapse of major financial "
        "institutions and a deep international recession. The crisis began with the subprime mortgage market. "
        "In the early 2000s, low interest rates and a push for homeownership led banks to issue high-risk "
        "loans to borrowers with poor credit."
    )

    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=60,
            min_length=20,
            num_beams=4,
            early_stopping=True,
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("Original Length (chars):", len(text))
    print("Summary:\n", summary)


if __name__ == "__main__":
    main()
