# Closed-Domain Question Answering API

A PyTorch GRU-based classifier that answers questions from a small, curated knowledge base. The trained model is exposed through a FastAPI service with validated requests, confidence-aware responses, health checks, automated tests, and Docker support.

## Why this is closed-domain QA

The dataset contains 90 question-answer examples and 85 unique answer labels. This model selects from those known answers; it is not a retrieval-augmented or open-domain language model. That constraint is intentional and is surfaced by the API's `supported` flag when confidence or vocabulary coverage is too low.

## Architecture

`CSV dataset -> tokenizer/vocabulary -> GRU classifier -> saved PyTorch artifact -> FastAPI /predict`

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn qa_service.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## API examples

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the capital of France?"}'
```

Example response:

```json
{
  "answer": "Paris",
  "confidence": 0.9987,
  "known_token_ratio": 1.0,
  "supported": true
}
```

## Test and containerize

```bash
pytest
docker build -t qa-api .
docker run -p 8000:8000 qa-api
```

## Resume-ready description

- Built a closed-domain question-answering service using PyTorch, a GRU text classifier, and a curated 90-example knowledge base.
- Designed and deployed FastAPI inference and health-check endpoints with Pydantic validation and confidence-based fallback handling.
- Created a reproducible training pipeline with serialized model artifacts, automated API tests, Docker packaging, and interactive OpenAPI documentation.
