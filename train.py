"""Train the QA model and create the API artifact."""

from pathlib import Path

from qa_service.training import train_and_save


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    metrics = train_and_save(root / "100_Unique_QA_Dataset.csv", root / "artifacts" / "qa_model.pt")
    print("Training complete")
    for name, value in metrics.items():
        print(f"{name}: {value}")
