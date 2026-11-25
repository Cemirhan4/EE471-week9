"""
EE471 Week 9 — Task 1.10: Automatic Speech Recognition (ASR)
Uses OpenAI's openai/whisper-large-v3 model for speech-to-text.
Loads WAV directly via soundfile/wave so system ffmpeg is not required.
"""

import os
import wave
import urllib.request
import numpy as np
import torch
from transformers import pipeline


SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)
SAMPLE_AUDIO = os.path.join(SAMPLE_DIR, "jfk.wav")
SAMPLE_URL = "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"


def ensure_sample_audio():
    if not os.path.exists(SAMPLE_AUDIO):
        print(f"Downloading sample audio to {SAMPLE_AUDIO}...")
        urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_AUDIO)
    return SAMPLE_AUDIO


def load_audio_array(audio_path: str):
    """Loads audio file into float32 numpy array without requiring system ffmpeg."""
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return audio, sr
    except Exception:
        with wave.open(audio_path, "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            return audio, sr


def main():
    audio_path = ensure_sample_audio()
    audio_array, sr = load_audio_array(audio_path)
    device = 0 if torch.cuda.is_available() else -1

    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        device=device,
        model_kwargs={"low_cpu_mem_usage": True},
    )
    result = asr(
        {"raw": audio_array, "sampling_rate": sr},
        generate_kwargs={"num_beams": 1, "language": "en", "task": "transcribe"},
    )

    print(f"Audio: {audio_path}")
    print(f"Transcription: {result['text']}")


if __name__ == "__main__":
    main()
