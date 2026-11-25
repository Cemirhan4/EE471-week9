# EE471 Week #9 — HuggingFace Transformers, Gradio & Tokenizers

**Course:** EE471 Modern Software Development Practices and Technologies (IYTE)  
**Student:** Cemirhan (`Cemirhan4`)

## Project Structure

- `task1_pipelines/` — 10 standalone Python scripts demonstrating HuggingFace `pipeline()` tasks (Sentiment Analysis, Zero-Shot, Text Generation, Fill-Mask, NER, QA, Summarization, Translation, ViT Image Classification, Whisper ASR).
- `task3_oop_gradio/` — Object-Oriented Python module (`hf_tasks.py`) + 10-tab interactive Gradio Web UI (`app.py`).
- `task4_tokenizers_models/` — 5 standalone Python scripts demonstrating Chapter 2 concepts (Load & Save Model, Encoding & Decoding, Padding, Truncation, Special Tokens).

## Setup

```powershell
cd C:\Users\Hp\Desktop\EE471-week9
pip install -r requirements.txt
```

## Running Tasks

### Task 1 (Individual Pipeline Scripts)
```powershell
python task1_pipelines/01_sentiment_analysis.py
python task1_pipelines/02_zero_shot_classification.py
python task1_pipelines/03_text_generation.py
python task1_pipelines/04_fill_mask.py
python task1_pipelines/05_ner.py
python task1_pipelines/06_question_answering.py
python task1_pipelines/07_summarization.py
python task1_pipelines/08_translation.py
python task1_pipelines/09_image_classification.py
python task1_pipelines/10_asr_whisper.py
```

### Task 3 (OOP Module + Gradio App)
```powershell
python task3_oop_gradio/app.py
```

### Task 4 (Models & Tokenizers)
```powershell
python task4_tokenizers_models/01_load_and_save.py
python task4_tokenizers_models/02_encode_and_decode.py
python task4_tokenizers_models/03_padding.py
python task4_tokenizers_models/04_truncation.py
python task4_tokenizers_models/05_special_tokens.py
```
