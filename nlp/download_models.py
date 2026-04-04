from __future__ import annotations

import os
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _env(name: str, default: str) -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


def download_model(model_id: str, cache_dir: Path) -> None:
    print(f"Downloading {model_id} to {cache_dir} ...")
    AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache_dir))
    AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=str(cache_dir))
    print(f"Finished {model_id}")


def main() -> None:
    cache_dir = Path(_env("HF_MODEL_CACHE", ".models")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    finbert_model_id = _env("FINBERT_MODEL_ID", "ProsusAI/finbert")
    mnli_model_id = _env("MNLI_MODEL_ID", "FacebookAI/roberta-large-mnli")

    print(f"Using model cache: {cache_dir}")
    download_model(finbert_model_id, cache_dir)
    download_model(mnli_model_id, cache_dir)
    print("All NLP models downloaded successfully.")


if __name__ == "__main__":
    main()
