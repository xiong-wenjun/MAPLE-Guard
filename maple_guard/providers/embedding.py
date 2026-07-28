from __future__ import annotations

from typing import List, Optional


class AverageEmbedder:
    @staticmethod
    def average_embeddings(embeddings: List[List[float]]) -> List[float]:
        if not embeddings:
            return []
        dim = len(embeddings[0])
        sums = [0.0] * dim
        used = 0
        for emb in embeddings:
            if len(emb) != dim:
                continue
            used += 1
            for i, value in enumerate(emb):
                sums[i] += float(value)
        if used == 0:
            return []
        return [v / used for v in sums]

    @staticmethod
    def weighted_average_embeddings(embeddings: List[List[float]], weights: Optional[List[float]] = None) -> List[float]:
        if not embeddings:
            return []
        if not weights:
            return AverageEmbedder.average_embeddings(embeddings)
        dim = len(embeddings[0])
        sums = [0.0] * dim
        total = 0.0
        for emb, weight in zip(embeddings, weights):
            if len(emb) != dim:
                continue
            w = float(weight)
            total += w
            for i, value in enumerate(emb):
                sums[i] += float(value) * w
        if total == 0:
            return AverageEmbedder.average_embeddings(embeddings)
        return [v / total for v in sums]
