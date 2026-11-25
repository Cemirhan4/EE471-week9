"""
EE471 Week 9 — Task 1.9: Image Classification with Google ViT
Uses google/vit-base-patch16-224 for basic image classification.
"""

import os
import urllib.request
from PIL import Image
from transformers import pipeline


SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)
SAMPLE_IMG = os.path.join(SAMPLE_DIR, "cats.jpg")
SAMPLE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


def ensure_sample_image():
    if not os.path.exists(SAMPLE_IMG):
        print(f"Downloading sample image to {SAMPLE_IMG}...")
        urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_IMG)
    return SAMPLE_IMG


def main():
    img_path = ensure_sample_image()
    image = Image.open(img_path).convert("RGB")

    classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
    results = classifier(image, top_k=5)

    print(f"Image: {img_path}\nTop 5 Predictions:")
    for res in results:
        print(f"  {res['label']:30s}: {res['score']:.4f}")


if __name__ == "__main__":
    main()
