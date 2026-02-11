"""
Transformer-based signal extractor for implicit psychological cues.
Uses Hugging Face hosted inference endpoints instead of local models.
"""

import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


class TransformerSignals:
    """
    Transformer-based signal extractor for implicit psychological cues.
    """

    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN", "")
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}

        self.finbert_api_url = (
            "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
        )
        self.mnli_api_url = (
            "https://router.huggingface.co/hf-inference/models/FacebookAI/roberta-large-mnli"
        )

    def _query(self, api_url: str, payload: dict[str, Any]) -> Any:
        if not self.hf_token:
            return None

        try:
            response = requests.post(
                api_url,
                headers=self.headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            return None

    def _mnli_zero_shot(
        self, text: str, candidate_labels: list[str]
    ) -> dict[str, list[float] | list[str]]:
        payload = {
            "inputs": text,
            "parameters": {
                "candidate_labels": candidate_labels,
                "multi_label": True,
            },
        }
        result = self._query(self.mnli_api_url, payload)

        if isinstance(result, dict) and "labels" in result and "scores" in result:
            return result

        return {"labels": candidate_labels, "scores": [0.5 for _ in candidate_labels]}

    # Fear Sentiment
    def extract_fear_sentiment(self, text: str) -> float:
        """
        Returns fear intensity score [0,1]
        """
        result = self._query(self.finbert_api_url, {"inputs": text})

        predictions = result if isinstance(result, list) else []
        if predictions and isinstance(predictions[0], list):
            predictions = predictions[0]

        if not predictions:
            return 0.5

        top = max(
            (p for p in predictions if isinstance(p, dict)),
            key=lambda item: float(item.get("score", 0.0)),
            default={"label": "neutral", "score": 0.5},
        )

        label = str(top.get("label", "neutral")).strip().lower()
        score = float(top.get("score", 0.5))

        if label == "negative":
            return score
        if label == "positive":
            return 1 - score
        return 0.5

    # Analytical and Herding Marker, returns both
    def extract_analytical_and_herding_markers(self, text: str) -> tuple[float, float]:
        """
        Returns (analytical_marker, herding_marker), both in [0, 1].
        """
        candidate_labels = [
            "analytical, data-driven reasoning",
            "herding, crowd-following behavior",
        ]

        result = self._mnli_zero_shot(text, candidate_labels)
        label_scores = dict(zip(result["labels"], result["scores"]))

        analytical = float(label_scores.get(candidate_labels[0], 0.5))
        herding = float(label_scores.get(candidate_labels[1], 0.5))

        analytical = max(0.0, min(1.0, analytical))
        herding = max(0.0, min(1.0, herding))

        return analytical, herding

    # Uncertainty Detection
    def extract_uncertainty_score(self, text: str) -> float:
        """
        Detects uncertainty in [0, 1].
        """
        candidate_labels = [
            "uncertain, unsure, hedging language",
            "confident, certain language",
        ]

        result = self._mnli_zero_shot(text, candidate_labels)
        label_scores = dict(zip(result["labels"], result["scores"]))

        uncertainty = float(label_scores.get(candidate_labels[0], 0.5))
        return max(0.0, min(1.0, uncertainty))
