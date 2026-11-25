"""
EE471 Week 9 — Task 1.2: Zero-Shot Classification
Classifies the Berkshire sentence without task-specific fine-tuning.
"""

from transformers import pipeline


def main():
    classifier = pipeline("zero-shot-classification")

    sentence = "Berkshire keeps their cash reserves at an extremely high level."
    candidate_labels = ["finance", "business", "politics", "education", "sports"]

    result = classifier(sentence, candidate_labels=candidate_labels)
    print(f"Sequence: {result['sequence']}")
    for label, score in zip(result["labels"], result["scores"]):
        print(f"  {label:12s}: {score:.4f}")


if __name__ == "__main__":
    main()
