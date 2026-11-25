"""
EE471 Week 9 — Task 1.5: Named Entity Recognition (NER)
Extracts person name, organization, and location from the given sentence.
"""

from transformers import pipeline


def main():
    ner = pipeline("ner", aggregation_strategy="simple")

    sentence = (
        "I am Nate, a research assistant at Izmir Institute of Technology, "
        "and currently living and working in the beautiful city İzmir in Türkiye."
    )

    entities = ner(sentence)
    print(f"Text: {sentence}\n")
    for ent in entities:
        print(
            f"Entity: {ent['word']:30s} | Group: {ent['entity_group']:5s} | Score: {ent['score']:.4f}"
        )


if __name__ == "__main__":
    main()
