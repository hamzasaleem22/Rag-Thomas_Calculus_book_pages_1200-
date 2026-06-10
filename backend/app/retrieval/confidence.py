from dataclasses import dataclass
from typing import Optional

from app.config import settings


@dataclass
class ClaimConfidence:
    claim_text: str
    nli_score: float
    nli_label: str
    consistency_rate: float
    token_overlap: float
    entity_overlap: float
    composite_confidence: float
    abstain: bool = False


class ConfidenceScorer:
    def __init__(self):
        self.abstain_threshold = getattr(settings, 'confidence_abstain_threshold', 0.4)
        self.warn_threshold = getattr(settings, 'confidence_warn_threshold', 0.6)

    def score_claim(
        self,
        claim_text: str,
        nli_score: float,
        nli_label: str,
        consistency_rate: float = 1.0,
        token_overlap: float = 0.0,
        entity_overlap: float = 0.0,
        is_mathematical: bool = False,
    ) -> ClaimConfidence:
        # Normalize NLI score to [0, 1]
        if nli_label == "entailment":
            nli_norm = min(1.0, max(0.0, nli_score))
        elif nli_label == "contradiction":
            nli_norm = 0.0
        else:
            nli_norm = max(0.0, nli_score) * 0.5

        # Penalize missing consistency signal
        consistency_weight = 0.3
        if consistency_rate < 0.5:
            consistency_weight *= 0.5

        # Composite confidence
        composite = (
            0.5 * nli_norm +
            consistency_weight * consistency_rate +
            0.2 * ((token_overlap + entity_overlap) / 2)
        )

        # Boost mathematical claims with strong NLI support
        if is_mathematical and nli_label == "entailment" and nli_norm > 0.7:
            composite = min(1.0, composite * 1.1)

        composite = max(0.0, min(1.0, composite))
        abstain = composite < self.abstain_threshold

        return ClaimConfidence(
            claim_text=claim_text,
            nli_score=nli_score,
            nli_label=nli_label,
            consistency_rate=consistency_rate,
            token_overlap=token_overlap,
            entity_overlap=entity_overlap,
            composite_confidence=round(composite, 3),
            abstain=abstain,
        )
