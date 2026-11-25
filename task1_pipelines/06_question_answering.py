"""
EE471 Week 9 — Task 1.6: Question Answering (Validation of NER entities)
Validates the extracted person, organization, and city using AutoModelForQuestionAnswering.
(Compatible with both Transformers v4 and v5)
"""

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


def answer_question(question: str, context: str, tokenizer, model):
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    start_scores = outputs.start_logits[0]
    end_scores = outputs.end_logits[0]

    start_idx = int(torch.argmax(start_scores))
    end_idx = int(torch.argmax(end_scores))
    if end_idx < start_idx:
        end_idx = start_idx

    answer_ids = inputs["input_ids"][0][start_idx : end_idx + 1]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    start_prob = float(torch.softmax(start_scores, dim=0)[start_idx])
    end_prob = float(torch.softmax(end_scores, dim=0)[end_idx])
    confidence = (start_prob + end_prob) / 2.0
    return answer, confidence


def main():
    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    context = (
        "I am Nate, a research assistant at Izmir Institute of Technology, "
        "and currently living and working in the beautiful city İzmir in Türkiye."
    )

    questions = [
        "What is the person's name?",
        "Which company or institution do they work for?",
        "Which city do they live in?",
    ]

    print(f"Context: {context}\n")
    for q in questions:
        ans, score = answer_question(q, context, tokenizer, model)
        print(f"Q: {q}")
        print(f"A: {ans} (score: {score:.4f})\n")


if __name__ == "__main__":
    main()
