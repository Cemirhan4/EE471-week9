"""
EE471 Week 9 — Task 1.1: Sentiment Analysis
Applies sentiment analysis to the two sentences specified in the exercise.
"""

from transformers import pipeline


def main():
    classifier = pipeline("sentiment-analysis")

    sentences = [
        "I've been waiting for a modern software practices course my whole life.",
        "I hate software courses",
    ]

    results = classifier(sentences)
    for sentence, result in zip(sentences, results):
        print(f"Text  : {sentence}")
        print(f"Result: {result['label']} (score: {result['score']:.4f})\n")


if __name__ == "__main__":
    main()
