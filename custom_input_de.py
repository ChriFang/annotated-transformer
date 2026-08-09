"""Translate custom German sentences with the trained Transformer model."""

from pathlib import Path

import spacy
import torch

# Reuse the model definition and greedy decoder from the annotated tutorial.
from the_annotated_transformer import greedy_decode, make_model


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "multi30k_model_final.pt"
VOCAB_PATH = ROOT / "vocab.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_resources():
    """Load tokenizers, vocabularies, and the trained model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {VOCAB_PATH}")

    spacy_de = spacy.load("de_core_news_sm")
    vocab_src, vocab_tgt = torch.load(VOCAB_PATH, map_location="cpu")

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return spacy_de, vocab_src, vocab_tgt, model


def translate(text, spacy_de, vocab_src, vocab_tgt, model, max_len=72):
    """Translate one German sentence into English."""
    tokens = [token.text for token in spacy_de.tokenizer(text)]
    source_ids = [
        vocab_src["<s>"],
        *vocab_src(tokens),
        vocab_src["</s>"],
    ]

    src = torch.tensor(source_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    src_mask = (src != vocab_src["<blank>"]).unsqueeze(-2)

    with torch.no_grad():
        output_ids = greedy_decode(
            model,
            src,
            src_mask,
            max_len=max_len,
            start_symbol=vocab_tgt["<s>"],
        )[0]

    result = []
    for token_id in output_ids.tolist():
        token = vocab_tgt.get_itos()[token_id]
        if token == "</s>":
            break
        if token not in {"<s>", "<blank>"}:
            result.append(token)

    return " ".join(result)


def main():
    print(f"Using device: {DEVICE}")
    print("Loading model...")
    spacy_de, vocab_src, vocab_tgt, model = load_resources()
    print("Model loaded. Enter a German sentence; press Enter on an empty line to exit.")

    while True:
        text = input("\nGerman: ").strip()
        if not text:
            break
        print("English:", translate(text, spacy_de, vocab_src, vocab_tgt, model))


if __name__ == "__main__":
    main()
