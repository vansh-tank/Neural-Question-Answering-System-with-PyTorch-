"""Training and artifact-loading helpers for the QA service."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from qa_service.model import QuestionClassifier
from qa_service.text import build_vocabulary, text_to_indices


@dataclass
class ModelConfig:
    embedding_dim: int = 64
    hidden_dim: int = 96
    confidence_threshold: float = 0.60


class QuestionAnswerDataset(Dataset):
    def __init__(self, questions: list[str], labels: list[int], vocabulary: dict[str, int]) -> None:
        self.questions = [torch.tensor(text_to_indices(question, vocabulary), dtype=torch.long) for question in questions]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.questions[index], self.labels[index]


def _collate(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    questions, labels = zip(*batch)
    return pad_sequence(questions, batch_first=True, padding_value=0), torch.tensor(labels, dtype=torch.long)


def train_and_save(
    dataset_path: Path,
    artifact_path: Path,
    epochs: int = 250,
    batch_size: int = 16,
    learning_rate: float = 0.003,
    seed: int = 42,
) -> dict[str, float | int]:
    """Train on the supplied closed-domain dataset and write a portable artifact."""
    random.seed(seed)
    torch.manual_seed(seed)
    frame = pd.read_csv(dataset_path)
    required_columns = {"question", "answer"}
    if not required_columns.issubset(frame.columns):
        raise ValueError("Dataset must contain 'question' and 'answer' columns.")
    if frame.empty:
        raise ValueError("Dataset must contain at least one question-answer pair.")

    questions = frame["question"].astype(str).tolist()
    answers = frame["answer"].astype(str).tolist()
    vocabulary = build_vocabulary(questions)
    answer_labels = sorted(set(answers))
    answer_to_index = {answer: index for index, answer in enumerate(answer_labels)}
    labels = [answer_to_index[answer] for answer in answers]

    dataset = QuestionAnswerDataset(questions, labels, vocabulary)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    config = ModelConfig()
    model = QuestionClassifier(len(vocabulary), len(answer_labels), config.embedding_dim, config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    final_loss = 0.0
    for _ in range(epochs):
        total_loss = 0.0
        for batch_questions, batch_labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_questions), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_labels)
        final_loss = total_loss / len(dataset)

    model.eval()
    correct = 0
    with torch.inference_mode():
        for batch_questions, batch_labels in loader:
            correct += (model(batch_questions).argmax(dim=1) == batch_labels).sum().item()

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocabulary": vocabulary,
            "answer_labels": answer_labels,
            "model_config": asdict(config),
            "training_metrics": {
                "train_examples": len(dataset),
                "unique_answers": len(answer_labels),
                "training_accuracy": correct / len(dataset),
                "final_loss": final_loss,
            },
        },
        artifact_path,
    )
    return {"train_examples": len(dataset), "unique_answers": len(answer_labels), "training_accuracy": correct / len(dataset), "final_loss": final_loss}


def load_model(artifact_path: Path) -> tuple[QuestionClassifier, dict[str, int], list[str], ModelConfig, dict]:
    """Load a model artifact produced by :func:`train_and_save`."""
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    config = ModelConfig(**artifact["model_config"])
    model = QuestionClassifier(
        vocabulary_size=len(artifact["vocabulary"]),
        answer_count=len(artifact["answer_labels"]),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    )
    model.load_state_dict(artifact["model_state"])
    model.eval()
    return model, artifact["vocabulary"], artifact["answer_labels"], config, artifact["training_metrics"]
