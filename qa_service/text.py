"""Text normalization shared by model training and inference."""

import re


def tokenize(text: str) -> list[str]:
    """Normalize text into simple word-like tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocabulary(questions: list[str]) -> dict[str, int]:
    """Build a vocabulary with explicit padding and unknown-token entries."""
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    for question in questions:
        for token in tokenize(question):
            vocabulary.setdefault(token, len(vocabulary))
    return vocabulary


def text_to_indices(text: str, vocabulary: dict[str, int]) -> list[int]:
    """Convert a question to vocabulary indices, retaining an unknown token."""
    tokens = tokenize(text)
    return [vocabulary.get(token, vocabulary["<UNK>"]) for token in tokens] or [vocabulary["<UNK>"]]
