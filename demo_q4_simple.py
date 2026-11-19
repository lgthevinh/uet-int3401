"""
Simple Q4 demo: type a word, get the predicted language.
Put this file next to models.py / train.py / backend.py. Do NOT submit it.
Run:  python demo_q4_simple.py
"""
import os

import torch
from torch.nn.functional import softmax

import models
import train
import backend

ALPHABET = "abcdefghijklmnopqrstuvwxyzáäèéêëíïñóöúüąćęłńśźż"   # 47 characters
LANGS = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
WEIGHTS = "q4_demo_weights.pt"


def get_model():
    model = models.LanguageIDModel()
    if os.path.exists(WEIGHTS):
        model.load_state_dict(torch.load(WEIGHTS))
    else:
        print("No saved weights, training once (a few minutes)...")
        dataset = backend.LanguageIDDataset(model)
        train.train_languageid(model, dataset)
        torch.save(model.state_dict(), WEIGHTS)
    model.eval()
    return model


def predict(model, word):
    # one-hot encode each character -> list of (1 x 47) tensors
    xs = []
    for ch in word:
        v = torch.zeros(1, len(ALPHABET))
        v[0, ALPHABET.index(ch)] = 1.0
        xs.append(v)
    with torch.no_grad():
        probs = softmax(model(xs), dim=1)[0]
    best = int(probs.argmax())
    return LANGS[best], probs[best].item()


def main():
    model = get_model()
    print("Type a word (empty line to quit).")
    while True:
        word = input("word> ").strip().lower()
        if not word:
            break
        bad = [c for c in word if c not in ALPHABET]
        if bad:
            print("  unsupported characters:", bad)
            continue
        lang, conf = predict(model, word)
        print(f"  -> {lang} ({conf * 100:.1f}%)")


if __name__ == "__main__":
    main()