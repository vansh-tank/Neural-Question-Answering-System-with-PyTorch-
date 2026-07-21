"""PyTorch model used by the QA classifier."""

import torch
from torch import nn


class QuestionClassifier(nn.Module):
    """Classifies a tokenized question into one of the known answer labels."""

    def __init__(
        self,
        vocabulary_size: int,
        answer_count: int,
        embedding_dim: int = 64,
        hidden_dim: int = 96,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, answer_count)

    def forward(self, question: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(question)
        _, hidden = self.rnn(embedded)
        return self.classifier(hidden[-1])
