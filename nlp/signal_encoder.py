"""
Transformer-backed signal extractor for implicit psychological cues.

Backend priority:
1. Local/offline Hugging Face models if available.
2. Hugging Face hosted inference endpoints if HF_TOKEN is configured.
3. Keyword heuristics as a last-resort fallback.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
    )
except Exception:  # pragma: no cover - optional dependency
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    pipeline = None


load_dotenv()


def _safe_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


class TransformerSignals:
    """
    Transformer-based signal extractor for implicit psychological cues.
    """

    def __init__(self):
        self.hf_token = _safe_env("HF_TOKEN")
        self.headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}

        self.backend_preference = _safe_env("NLP_BACKEND", "auto").lower()
        self.model_cache_dir = _safe_env("HF_MODEL_CACHE") or _safe_env("TRANSFORMERS_CACHE")

        self.finbert_model_id = _safe_env("FINBERT_MODEL_ID", "ProsusAI/finbert")
        self.mnli_model_id = _safe_env("MNLI_MODEL_ID", "FacebookAI/roberta-large-mnli")

        self.finbert_api_url = (
            f"https://router.huggingface.co/hf-inference/models/{self.finbert_model_id}"
        )
        self.mnli_api_url = (
            f"https://router.huggingface.co/hf-inference/models/{self.mnli_model_id}"
        )

        self._local_sentiment = None
        self._local_zero_shot = None
        self._local_backend_ready = False
        self._last_backend = "heuristic"
        self._local_backend_error: str | None = None

        self._initialize_local_backend()

    def _initialize_local_backend(self) -> None:
        if self.backend_preference == "remote":
            return
        if pipeline is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            self._local_backend_error = "transformers_not_installed"
            return

        try:
            finbert_kwargs = {}
            mnli_kwargs = {}
            if self.model_cache_dir:
                finbert_kwargs["cache_dir"] = self.model_cache_dir
                mnli_kwargs["cache_dir"] = self.model_cache_dir

            finbert_tokenizer = AutoTokenizer.from_pretrained(self.finbert_model_id, **finbert_kwargs)
            finbert_model = AutoModelForSequenceClassification.from_pretrained(self.finbert_model_id, **finbert_kwargs)
            mnli_tokenizer = AutoTokenizer.from_pretrained(self.mnli_model_id, **mnli_kwargs)
            mnli_model = AutoModelForSequenceClassification.from_pretrained(self.mnli_model_id, **mnli_kwargs)

            self._local_sentiment = pipeline(
                "text-classification",
                model=finbert_model,
                tokenizer=finbert_tokenizer,
            )
            self._local_zero_shot = pipeline(
                "zero-shot-classification",
                model=mnli_model,
                tokenizer=mnli_tokenizer,
            )
            self._local_backend_ready = True
        except Exception as exc:  # pragma: no cover - depends on local model state
            self._local_backend_ready = False
            self._local_backend_error = str(exc)

    def backend_mode(self) -> str:
        if self._local_backend_ready:
            return "local_transformers"
        if self.hf_token:
            return "remote_hf_inference"
        return "keyword_heuristics"

    def last_backend(self) -> str:
        return self._last_backend

    def has_local_support(self) -> bool:
        return self._local_backend_ready

    def has_remote_support(self) -> bool:
        return bool(self.hf_token)

    def _keyword_ratio(self, text: str, keywords: list[str]) -> float:
        text_lower = text.lower()
        hits = sum(1 for keyword in keywords if keyword in text_lower)
        if not keywords:
            return 0.0
        return max(0.0, min(1.0, hits / max(len(keywords) * 0.35, 1.0)))

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
            self._last_backend = "remote_hf_inference"
            return response.json()
        except requests.RequestException:
            return None

    def _local_sentiment_query(self, text: str) -> Any:
        if not self._local_backend_ready or self._local_sentiment is None:
            return None
        try:
            result = self._local_sentiment(text)
            self._last_backend = "local_transformers"
            return result
        except Exception:
            return None

    def _local_zero_shot_query(self, text: str, candidate_labels: list[str]) -> Any:
        if not self._local_backend_ready or self._local_zero_shot is None:
            return None
        try:
            result = self._local_zero_shot(
                text,
                candidate_labels=candidate_labels,
                multi_label=True,
            )
            self._last_backend = "local_transformers"
            return result
        except Exception:
            return None

    def _mnli_zero_shot(
        self, text: str, candidate_labels: list[str]
    ) -> dict[str, list[float] | list[str]]:
        result = None
        if self.backend_preference != "heuristic":
            if self.backend_preference in ("auto", "local"):
                result = self._local_zero_shot_query(text, candidate_labels)
            if result is None and self.backend_preference in ("auto", "remote"):
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

        self._last_backend = "keyword_heuristics"
        return {"labels": candidate_labels, "scores": [0.5 for _ in candidate_labels]}

    def _score_label_pair(
        self,
        text: str,
        positive_label: str,
        negative_label: str,
    ) -> tuple[float, float]:
        result = self._mnli_zero_shot(text, [positive_label, negative_label])
        label_scores = dict(zip(result["labels"], result["scores"]))
        positive = float(label_scores.get(positive_label, 0.5))
        negative = float(label_scores.get(negative_label, 0.5))
        positive = max(0.0, min(1.0, positive))
        negative = max(0.0, min(1.0, negative))
        return positive, negative

    def _is_neutral_pair(self, positive: float, negative: float) -> bool:
        return abs(positive - 0.5) < 1e-9 and abs(negative - 0.5) < 1e-9

    def extract_fear_sentiment(self, text: str) -> float:
        result = None
        if self.backend_preference != "heuristic":
            if self.backend_preference in ("auto", "local"):
                result = self._local_sentiment_query(text)
            if result is None and self.backend_preference in ("auto", "remote"):
                result = self._query(self.finbert_api_url, {"inputs": text})

        predictions = result if isinstance(result, list) else []
        if predictions and isinstance(predictions[0], list):
            predictions = predictions[0]

        if not predictions:
            self._last_backend = "keyword_heuristics"
            return max(
                0.5,
                self._keyword_ratio(
                    text,
                    [
                        "scared",
                        "worried",
                        "fear",
                        "afraid",
                        "crash",
                        "losing money",
                        "loss",
                        "downside",
                        "panic",
                        "market swings",
                    ],
                ),
            )

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

    def extract_analytical_and_herding_markers(self, text: str) -> tuple[float, float]:
        candidate_labels = [
            "analytical, data-driven reasoning",
            "herding, crowd-following behavior",
        ]

        result = self._mnli_zero_shot(text, candidate_labels)
        label_scores = dict(zip(result["labels"], result["scores"]))

        analytical = float(label_scores.get(candidate_labels[0], 0.5))
        herding = float(label_scores.get(candidate_labels[1], 0.5))

        if self._is_neutral_pair(analytical, herding):
            self._last_backend = "keyword_heuristics"
            analytical_ratio = self._keyword_ratio(
                text,
                [
                    "financial statements",
                    "data",
                    "analysis",
                    "structured reasoning",
                    "structured research",
                    "probability estimates",
                    "scenario analysis",
                    "earnings",
                    "valuation",
                    "cash flow",
                    "balance sheet quality",
                    "trend",
                    "research",
                    "balance sheet",
                    "evidence",
                    "checklists",
                ],
            )
            herding_ratio = self._keyword_ratio(
                text,
                [
                    "everyone is buying",
                    "everyone else",
                    "crowd",
                    "fomo",
                    "trending",
                    "people are buying",
                    "social media",
                    "popular stock",
                ],
            )
            analytical = max(analytical, 0.55 if analytical_ratio > 0 else 0.5, analytical_ratio)
            herding = max(herding, 0.55 if herding_ratio > 0 else 0.5, herding_ratio)

        analytical = max(0.0, min(1.0, analytical))
        herding = max(0.0, min(1.0, herding))
        return analytical, herding

    def extract_uncertainty_score(self, text: str) -> float:
        candidate_labels = [
            "uncertain, unsure, hedging language",
            "confident, certain language",
        ]

        result = self._mnli_zero_shot(text, candidate_labels)
        label_scores = dict(zip(result["labels"], result["scores"]))

        uncertainty = float(label_scores.get(candidate_labels[0], 0.5))
        if abs(uncertainty - 0.5) < 1e-9:
            self._last_backend = "keyword_heuristics"
            uncertainty_ratio = self._keyword_ratio(
                text,
                [
                    "not sure",
                    "uncertain",
                    "unsure",
                    "maybe",
                    "confused",
                    "hard to decide",
                    "i don't know",
                    "hesitant",
                ],
            )
            uncertainty = max(uncertainty, 0.55 if uncertainty_ratio > 0 else 0.5, uncertainty_ratio)
        return max(0.0, min(1.0, uncertainty))

    def extract_risk_language_density(self, text: str) -> float:
        risk, _ = self._score_label_pair(
            text,
            "risk-focused, loss-aware, downside-protection language",
            "growth-focused, upside-seeking, return-maximizing language",
        )
        if self._is_neutral_pair(risk, 0.5):
            self._last_backend = "keyword_heuristics"
            risk = max(
                0.5,
                self._keyword_ratio(
                    text,
                    [
                        "risk",
                        "loss",
                        "downside",
                        "drawdown",
                        "capital preservation",
                        "protect",
                        "market swings",
                    ],
                ),
            )
        return risk

    def extract_urgency_score(self, text: str) -> float:
        urgency, _ = self._score_label_pair(
            text,
            "urgent, immediate-action, short-fuse decision making",
            "patient, deliberate, willing-to-wait decision making",
        )
        if self._is_neutral_pair(urgency, 0.5):
            self._last_backend = "keyword_heuristics"
            urgency = max(
                0.5,
                self._keyword_ratio(
                    text,
                    [
                        "right now",
                        "immediately",
                        "immediate move",
                        "this week",
                        "quick",
                        "urgent",
                    ],
                ),
            )
        return urgency

    def extract_time_horizon_bias(self, text: str) -> float:
        long_term, short_term = self._score_label_pair(
            text,
            "long-term investing, compounding, multi-year holding period",
            "short-term trading, quick return, near-term exit focus",
        )
        if self._is_neutral_pair(long_term, short_term):
            self._last_backend = "keyword_heuristics"
            long_term = self._keyword_ratio(
                text,
                [
                    "long term",
                    "holding for years",
                    "compounding",
                    "multi-year",
                    "for months",
                ],
            )
            short_term = self._keyword_ratio(
                text,
                [
                    "this week",
                    "right now",
                    "quick return",
                    "immediate move",
                    "short term",
                ],
            )
        return max(-1.0, min(1.0, long_term - short_term))

    def extract_locus_of_control(self, text: str) -> tuple[float, float]:
        internal, external = self._score_label_pair(
            text,
            "internal locus of control, personal agency, disciplined plan",
            "external locus of control, market-driven, fate-driven helplessness",
        )
        if self._is_neutral_pair(internal, external):
            self._last_backend = "keyword_heuristics"
            internal = max(
                0.5,
                self._keyword_ratio(
                    text,
                    [
                        "my plan",
                        "my strategy",
                        "trust my own plan",
                        "fully in control",
                        "stick to my process",
                        "personal agency",
                        "follow my own process",
                        "my own discipline",
                        "following my own plan",
                    ],
                ),
            )
            external = max(
                0.5,
                self._keyword_ratio(
                    text,
                    [
                        "market decides for me",
                        "forcing my hand",
                        "out of my control",
                        "luck",
                        "fate",
                        "pushing me around",
                        "outside forces",
                        "externally driven",
                        "helpless",
                        "market noise",
                    ],
                ),
            )
        return internal, external

    def diagnostics(self) -> dict[str, object]:
        return {
            "backend_preference": self.backend_preference,
            "backend_mode": self.backend_mode(),
            "last_backend": self._last_backend,
            "local_backend_ready": self._local_backend_ready,
            "has_remote_support": self.has_remote_support(),
            "model_cache_dir": self.model_cache_dir or None,
            "finbert_model_id": self.finbert_model_id,
            "mnli_model_id": self.mnli_model_id,
            "local_backend_error": self._local_backend_error,
        }


def default_local_model_root() -> Path:
    return Path(os.getenv("HF_MODEL_CACHE", ".models")).resolve()
