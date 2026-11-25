"""
EE471 Week 9 — Task 3: Object-Oriented HuggingFace NLP/Vision/Audio Module
Encapsulates all 10 HuggingFace tasks in a clean OOP class with lazy caching.
Compatible with Transformers v4 and v5.
"""

import torch
from PIL import Image
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
)


class HuggingFaceTasks:
    """
    Unified OOP wrapper around HuggingFace Transformers
    for all 10 tasks defined in EE471 Week 9.
    """

    def __init__(self, device: int = None):
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device
        self._pipelines = {}
        self._models = {}

    def _get_pipeline(self, key: str, task: str, model: str = None, **kwargs):
        """Lazily loads and caches a HuggingFace pipeline."""
        if key not in self._pipelines:
            pipe_kwargs = {"device": self.device, **kwargs}
            if model:
                pipe_kwargs["model"] = model
            self._pipelines[key] = pipeline(task, **pipe_kwargs)
        return self._pipelines[key]

    def _get_seq2seq(self, key: str, model_name: str):
        """Lazily loads and caches a Seq2Seq tokenizer and model."""
        if key not in self._models:
            tok = AutoTokenizer.from_pretrained(model_name)
            mod = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self._models[key] = (tok, mod)
        return self._models[key]

    def sentiment_analysis(self, text: str) -> dict:
        """Task 1: Classify text sentiment (POSITIVE / NEGATIVE)."""
        clf = self._get_pipeline("sentiment", "sentiment-analysis")
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        results = clf(lines)
        return {
            f"Sentence {i+1} ({r['label']})": round(float(r["score"]), 4)
            for i, r in enumerate(results)
        }

    def zero_shot_classification(self, text: str, labels: str) -> dict:
        """Task 2: Classify text into arbitrary candidate labels without fine-tuning."""
        clf = self._get_pipeline("zero_shot", "zero-shot-classification")
        candidate_labels = [lbl.strip() for lbl in labels.split(",") if lbl.strip()]
        res = clf(text, candidate_labels=candidate_labels)
        return {
            label: round(float(score), 4)
            for label, score in zip(res["labels"], res["scores"])
        }

    def text_generation(self, prompt: str, max_new_tokens: int = 35, num_return_sequences: int = 2) -> str:
        """Task 3: Generate continuations from an incomplete prompt."""
        gen = self._get_pipeline("generation", "text-generation", model="gpt2")
        outputs = gen(
            prompt,
            max_new_tokens=int(max_new_tokens),
            num_return_sequences=int(num_return_sequences),
            do_sample=True,
            pad_token_id=50256,
        )
        formatted = []
        for idx, out in enumerate(outputs, 1):
            words = out["generated_text"].split()[:35]
            formatted.append(f"Alternative {idx}:\n" + " ".join(words))
        return "\n\n".join(formatted)

    def mask_filling(self, sentence: str, top_k: int = 3) -> dict:
        """Task 4: Predict masked tokens in a sentence (use <mask> or [MASK])."""
        unmasker = self._get_pipeline("fill_mask", "fill-mask", model="distilroberta-base")
        mask_token = unmasker.tokenizer.mask_token
        normalized = sentence.replace("[MASK]", mask_token).replace("[…]", mask_token)
        results = unmasker(normalized, top_k=int(top_k))
        return {
            f"{res['token_str'].strip()} -> {res['sequence']}": round(float(res["score"]), 4)
            for res in results
        }

    def named_entity_recognition(self, text: str) -> str:
        """Task 5: Extract Person, Organization, and Location entities."""
        ner = self._get_pipeline("ner", "ner", aggregation_strategy="simple")
        entities = ner(text)
        if not entities:
            return "No entities detected."
        lines = [
            f"- {ent['word']}  |  Group: {ent['entity_group']}  |  Score: {ent['score']:.4f}"
            for ent in entities
        ]
        return "\n".join(lines)

    def question_answering(self, context: str, question: str) -> str:
        """Task 6: Answer a question given a context paragraph."""
        if "qa" not in self._models:
            model_name = "distilbert-base-cased-distilled-squad"
            self._models["qa"] = (
                AutoTokenizer.from_pretrained(model_name),
                AutoModelForQuestionAnswering.from_pretrained(model_name),
            )
        tokenizer, model = self._models["qa"]
        inputs = tokenizer(question, context, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        start_idx = int(torch.argmax(outputs.start_logits[0]))
        end_idx = max(start_idx, int(torch.argmax(outputs.end_logits[0])))
        answer_ids = inputs["input_ids"][0][start_idx : end_idx + 1]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
        start_p = float(torch.softmax(outputs.start_logits[0], dim=0)[start_idx])
        end_p = float(torch.softmax(outputs.end_logits[0], dim=0)[end_idx])
        return f"Answer: {answer}\nConfidence: {(start_p + end_p) / 2.0:.4f}"

    def text_summarization(self, text: str, max_length: int = 60, min_length: int = 20) -> str:
        """Task 7: Summarize long input text."""
        tokenizer, model = self._get_seq2seq("summarization", "sshleifer/distilbart-cnn-12-6")
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        with torch.no_grad():
            ids = model.generate(
                inputs["input_ids"],
                max_length=int(max_length),
                min_length=int(min_length),
                num_beams=4,
                early_stopping=True,
            )
        return tokenizer.decode(ids[0], skip_special_tokens=True)

    def text_translation(self, text: str) -> str:
        """Task 8: Translate French text into English."""
        tokenizer, model = self._get_seq2seq("translation", "Helsinki-NLP/opus-mt-fr-en")
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(**inputs)
        return tokenizer.decode(ids[0], skip_special_tokens=True)

    def image_classification(self, image: Image.Image) -> dict:
        """Task 9: Classify an image using Google ViT (google/vit-base-patch16-224)."""
        if image is None:
            return {"Error": 0.0}
        clf = self._get_pipeline("image_clf", "image-classification", model="google/vit-base-patch16-224")
        results = clf(image.convert("RGB"), top_k=5)
        return {res["label"]: round(float(res["score"]), 4) for res in results}

    def automatic_speech_recognition(self, audio_path: str) -> str:
        """Task 10: Transcribe speech audio using OpenAI Whisper (no ffmpeg needed for WAV/FLAC)."""
        if not audio_path:
            return "Please upload or record an audio file."
        import numpy as np
        import wave
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
        except Exception:
            with wave.open(audio_path, "rb") as wf:
                sr = wf.getframerate()
                channels = wf.getnchannels()
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                if channels > 1:
                    audio = audio.reshape(-1, channels).mean(axis=1)
        asr = self._get_pipeline(
            "asr",
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            model_kwargs={"low_cpu_mem_usage": True},
        )
        res = asr(
            {"raw": audio, "sampling_rate": sr},
            generate_kwargs={"num_beams": 1, "language": "en", "task": "transcribe"},
        )
        return res["text"]
