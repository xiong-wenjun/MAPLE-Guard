from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseLLM(ABC):
    def __init__(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        pass

    @abstractmethod
    def extract_keywords(self, text: str, max_keywords: int = 8) -> List[str]:
        pass

    def generate_script(self, trajectory: str) -> str:
        prompt = (
            "Analyze the following task trajectory and generate a concise, high-level script "
            "with 3-5 reusable steps.\n\n"
            f"{trajectory}"
        )
        return self.generate([{"role": "user", "content": prompt}], temperature=0)


class BaseEmbedder(ABC):
    def __init__(self, max_text_len: int = 8196, **kwargs: Any) -> None:
        self.max_text_len = max_text_len

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        pass

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]
