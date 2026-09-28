"""Inference adapter for the official pinned Hugging Face PIGuard release."""
from __future__ import annotations

from collections.abc import Sequence
import math

MODEL_ID = "leolee99/PIGuard"
MODEL_REVISION = "dd78b24e330193a22d2293ac66922dd4f982f563"
SOURCE_REVISION = "1b5751e88bf7475acbedfc8eda795ce060307c84"


class PIGuardDetector:
    """Return injection probability from PIGuard's official custom model head."""

    def __init__(self, args, *, tokenizer_loader=None, model_loader=None):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PIGuard requires PyTorch; no fallback detector is available") from exc
        self._torch = torch
        self.model_id = getattr(args, "piguard_model", MODEL_ID)
        self.revision = getattr(args, "piguard_revision", MODEL_REVISION)
        self.device = getattr(args, "piguard_device", "cpu")
        self.max_length = getattr(args, "piguard_max_length", 512)
        self.threshold = getattr(args, "piguard_threshold", 0.5)
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("PIGuard model must be a nonempty model ID or local directory")
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("PIGuard revision must be specified explicitly")
        if type(self.max_length) is not int or self.max_length <= 0:
            raise ValueError("PIGuard max_length must be a positive integer")
        if (type(self.threshold) not in (int, float) or not math.isfinite(self.threshold)
                or not 0 <= self.threshold <= 1):
            raise ValueError("PIGuard threshold must be finite and within [0, 1]")
        self._injected_loaders = tokenizer_loader is not None or model_loader is not None
        self._counters = {"calls": 0, "model_calls": 0, "texts_scored": 0, "errors": 0}
        try:
            if tokenizer_loader is None or model_loader is None:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                tokenizer_loader = tokenizer_loader or AutoTokenizer.from_pretrained
                model_loader = model_loader or AutoModelForSequenceClassification.from_pretrained
            self.tokenizer = tokenizer_loader(self.model_id, revision=self.revision,
                                              trust_remote_code=True, use_fast=True)
            self.model = model_loader(self.model_id, revision=self.revision,
                                      trust_remote_code=True, torch_dtype=torch.float32)
        except Exception as exc:
            raise RuntimeError(
                f"PIGuard loading failed for {self.model_id}@{self.revision}; "
                "install the official dependencies and make the pinned checkpoint available. "
                "No fallback detector was used"
            ) from exc
        config = self.model.config
        if config.num_labels != 2 or config.id2label != {0: "benign", 1: "injection"}:
            raise ValueError("PIGuard requires exact labels 0=benign, 1=injection")
        try:
            self.model = self.model.to(self.device).float().eval()
        except Exception as exc:
            raise RuntimeError(f"PIGuard could not initialize FP32 evaluation on {self.device}") from exc

    def score(self, text: str) -> float:
        return self.score_batch([text])[0]

    def score_batch(self, texts) -> list[float]:
        self._counters["calls"] += 1
        try:
            if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
                raise TypeError("score_batch expects a sequence of text strings")
            if any(not isinstance(text, str) for text in texts):
                raise TypeError("Every PIGuard input must be a string")
            if not texts:
                return []
            texts = list(texts)
            inputs = self.tokenizer(texts, return_tensors="pt", truncation=True,
                                    padding=True, max_length=self.max_length)
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            with self._torch.inference_mode():
                self._counters["model_calls"] += 1
                logits = self.model(**inputs).logits.float()
                if tuple(logits.shape) != (len(texts), 2):
                    raise ValueError("Expected one pair of logits per input text")
                if not self._torch.isfinite(logits).all().item():
                    raise ValueError("Model returned non-finite logits")
                scores = self._torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            self._counters["texts_scored"] += len(texts)
            return scores
        except Exception as exc:
            self._counters["errors"] += 1
            raise RuntimeError("PIGuard inference failed; no fallback score was used") from exc

    def provenance(self) -> dict:
        """JSON-compatible runtime configuration and completed-inference counters."""
        return {
            "detector": "PIGuard",
            "source_url": "https://github.com/leolee99/PIGuard",
            "source_revision": SOURCE_REVISION,
            "model": self.model_id,
            "model_revision": self.revision,
            "tokenizer_revision": self.revision,
            "official_model_requested": self.model_id == MODEL_ID and self.revision == MODEL_REVISION,
            "profile": ("official_hf_inference_examples_512" if self.max_length == 512
                        else "custom_max_length"),
            "id2label": {"0": "benign", "1": "injection"},
            "score": "softmax(logits, dim=-1)[:, 1]",
            "threshold": self.threshold,
            "decision_rule": "block if injection_probability > threshold; ties allowed",
            "max_length": self.max_length,
            "truncation": True,
            "padding": "longest_in_batch",
            "use_fast": True,
            "device": str(self.device),
            "dtype": "float32",
            "trust_remote_code": True,
            "injected_loaders": self._injected_loaders,
            "chunking": False,
            "fallback": False,
            "counters": dict(self._counters),
        }
