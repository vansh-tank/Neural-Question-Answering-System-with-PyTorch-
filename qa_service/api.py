"""FastAPI application for serving the trained QA classifier."""

from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from qa_service.text import text_to_indices
from qa_service.training import load_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "qa_model.pt"


class QuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=300, examples=["What is the capital of France?"])


class PredictionResponse(BaseModel):
    answer: str | None
    confidence: float
    known_token_ratio: float
    supported: bool


def create_app(artifact_path: Path = DEFAULT_ARTIFACT_PATH) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not artifact_path.exists():
            raise RuntimeError(f"Model artifact not found at {artifact_path}. Run `python train.py` first.")
        model, vocabulary, answers, config, metrics = load_model(artifact_path)
        app.state.model = model
        app.state.vocabulary = vocabulary
        app.state.answers = answers
        app.state.confidence_threshold = config.confidence_threshold
        app.state.metrics = metrics
        yield

    app = FastAPI(
        title="Closed-Domain QA API",
        version="1.0.0",
        description="A PyTorch GRU classifier served through FastAPI.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": "loaded", "metrics": app.state.metrics}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: QuestionRequest) -> PredictionResponse:
        question_indices = text_to_indices(request.question, app.state.vocabulary)
        question_tensor = torch.tensor(question_indices, dtype=torch.long).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(app.state.model(question_tensor), dim=1)[0]
        confidence, label_index = torch.max(probabilities, dim=0)
        confidence_value = round(float(confidence.item()), 4)
        known_tokens = sum(index != app.state.vocabulary["<UNK>"] for index in question_indices)
        known_token_ratio = round(known_tokens / len(question_indices), 4)
        # A closed-domain model should not confidently invent an answer for entirely unfamiliar input.
        supported = (
            confidence_value >= app.state.confidence_threshold
            and known_token_ratio >= 0.5
        )
        return PredictionResponse(
            answer=app.state.answers[label_index.item()] if supported else None,
            confidence=confidence_value,
            known_token_ratio=known_token_ratio,
            supported=supported,
        )

    return app


app = create_app()
