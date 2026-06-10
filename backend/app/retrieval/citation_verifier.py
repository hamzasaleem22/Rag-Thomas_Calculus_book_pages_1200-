import re
from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document
from app.config import settings


STOPWORDS = {
    'the', 'is', 'at', 'which', 'what', 'how', 'do', 'does', 'a', 'an',
    'and', 'or', 'of', 'to', 'for', 'in', 'on', 'by', 'with', 'from', 'its',
    'are', 'be', 'was', 'were', 'been', 'being', 'that', 'this', 'these',
    'those', 'it', 'we', 'they', 'not', 'no', 'but', 'all', 'each', 'every',
    'some', 'any', 'there', 'their', 'them', 'than', 'then', 'also', 'very',
    'so', 'if', 'as', 'when', 'where', 'why', 'who', 'whom', 'can', 'will',
    'may', 'just', 'have', 'has', 'had', 'do', 'does', 'did',
}


_nli_verifier: Optional[object] = None


def get_nli_verifier():
    global _nli_verifier
    if _nli_verifier is None and settings.use_nli_verifier:
        try:
            from app.retrieval.nli_verifier import NLIVerifier
            _nli_verifier = NLIVerifier(
                model_name=settings.nli_model_name,
                device=settings.nli_device,
                cache_max=settings.nli_cache_max_size,
            )
        except Exception as e:
            print(f"  Failed to load NLI verifier: {e}")
            _nli_verifier = None
    return _nli_verifier


def extract_claims_around_citation(text: str, tag: str) -> list[str]:
    parts = text.split(tag)
    claims = []
    for i in range(len(parts) - 1):
        before = parts[i].rsplit(". ", 1)[-1][-200:] if parts[i] else ""
        after = parts[i + 1][:200].split(". ")[0] if parts[i + 1] else ""
        claim = (before + " " + after).strip()
        if claim:
            claims.append(claim)
    return claims


def verify_token_overlap(claim: str, doc_text: str, threshold: float = 0.3) -> tuple[bool, float]:
    claim_terms = set(re.findall(r'\b[a-zA-Z]\w+\b', claim.lower())) - STOPWORDS
    doc_terms = set(re.findall(r'\b[a-zA-Z]\w+\b', doc_text.lower())) - STOPWORDS
    if not claim_terms:
        return True, 1.0
    overlap = len(claim_terms & doc_terms)
    score = overlap / len(claim_terms)
    return score >= threshold, score


def verify_entity_overlap(claim: str, doc_text: str, threshold: float = 0.5) -> tuple[bool, float]:
    entity_patterns = [
        r'\b[A-Z][A-Z]+\b',
        r'Theorem\s+\d+(?:\.\d+)?',
        r'Definition\s+\d+(?:\.\d+)?',
        r'EXAMPLE\s+\d+',
        r'Rule\s+\d+',
        r'\\[a-zA-Z]+',
        r'\b\d+\.\d+\b',
    ]
    claim_entities = set()
    for pat in entity_patterns:
        for m in re.finditer(pat, claim):
            claim_entities.add(m.group(0).lower())

    doc_entities = set()
    for pat in entity_patterns:
        for m in re.finditer(pat, doc_text):
            doc_entities.add(m.group(0).lower())

    if not claim_entities:
        return True, 1.0
    overlap = len(claim_entities & doc_entities)
    score = overlap / len(claim_entities) if claim_entities else 1.0
    return score >= threshold, score


def verify_key_terms(claim: str, doc_text: str, threshold: float = 0.3) -> tuple[bool, float]:
    doc_terms = set(re.findall(r'\b[a-zA-Z]\w+\b', doc_text.lower())) - STOPWORDS

    claim_phrases = re.findall(r'(?:the\s+)?[A-Za-z][A-Za-z\s]+(?:theorem|rule|law|formula|property|test)', claim, re.IGNORECASE)
    if not claim_phrases:
        claim_phrases = [claim]

    for phrase in claim_phrases:
        phrase_terms = set(re.findall(r'\b[a-zA-Z]\w+\b', phrase.lower())) - STOPWORDS
        if not phrase_terms:
            continue
        overlap = len(phrase_terms & doc_terms)
        score = overlap / len(phrase_terms)
        if score < threshold:
            return False, score
    return True, 1.0


def verify_claim_with_nli(claim: str, doc_text: str) -> tuple[float, str]:
    verifier = get_nli_verifier()
    if verifier is not None:
        try:
            return verifier.entailment_score(claim, doc_text)
        except Exception as e:
            print(f"  NLI verification failed: {e}")
    return 0.0, "neutral"


def verify_citation(claim: str, doc: Document) -> dict:
    doc_text = doc.page_content

    token_ok, token_score = verify_token_overlap(claim, doc_text)
    entity_ok, entity_score = verify_entity_overlap(claim, doc_text)
    key_terms_ok, key_terms_score = verify_key_terms(claim, doc_text)

    if settings.use_nli_verifier:
        nli_score, nli_label = verify_claim_with_nli(claim, doc_text)
        w = settings.nli_verification_weight
        composite_score = w * max(0, nli_score) + 0.2 * token_score + 0.15 * entity_score + 0.15 * key_terms_score
        nli_is_contradiction = (nli_label == "contradiction")
        verified = not nli_is_contradiction and composite_score >= settings.relevance_threshold
    else:
        nli_score = 0.0
        nli_label = "neutral"
        composite_score = 0.5 * token_score + 0.3 * entity_score + 0.2 * key_terms_score
        verified = token_ok and (entity_ok or key_terms_ok)

    return {
        "verified": verified,
        "composite_score": round(composite_score, 3),
        "token_overlap_score": round(token_score, 3),
        "entity_overlap_score": round(entity_score, 3),
        "key_terms_score": round(key_terms_score, 3),
        "nli_score": round(nli_score, 3),
        "nli_label": nli_label,
    }


def verify_all_citations(answer: str, documents: list[Document]) -> dict:
    results = {}
    all_verified = True
    total_score = 0.0
    citation_count = 0

    for i, doc in enumerate(documents):
        tag = f"[{i+1}]"
        if tag not in answer:
            continue
        claims = extract_claims_around_citation(answer, tag)
        doc_results = []
        for claim in claims:
            result = verify_citation(claim, doc)
            doc_results.append(result)
            if not result["verified"]:
                all_verified = False
            total_score += result["composite_score"]
            citation_count += 1
        results[i] = {"tag": tag, "claims": doc_results, "doc_page": doc.metadata.get("page")}

    return {
        "results": results,
        "all_verified": all_verified,
        "avg_score": round(total_score / max(citation_count, 1), 3),
        "citation_count": citation_count,
    }


def get_cited_documents(answer: str, documents: list[Document]) -> list[Document]:
    cited_indices = set()
    for doc_idx in range(1, len(documents) + 1):
        if f"[{doc_idx}]" in answer:
            cited_indices.add(doc_idx - 1)
    return [documents[i] for i in sorted(cited_indices)]


@dataclass
class ClaimVerificationResult:
    claim_text: str
    sentence_idx: int
    claim_idx: int
    is_mathematical: bool
    is_core_claim: bool
    nli_score: float
    nli_label: str
    token_overlap_score: float
    entity_overlap_score: float
    composite_score: float
    verified: bool
    source_doc_idx: Optional[int] = None
    source_doc_page: Optional[int] = None


_claim_extractor: Optional[object] = None


def get_claim_extractor():
    global _claim_extractor
    if _claim_extractor is None:
        try:
            from app.retrieval.claim_extractor import AtomicClaimExtractor
            _claim_extractor = AtomicClaimExtractor()
        except Exception as e:
            print(f"  Failed to load claim extractor: {e}")
            _claim_extractor = None
    return _claim_extractor


def verify_answer_claims(answer: str, documents: list[Document]) -> list[ClaimVerificationResult]:
    extractor = get_claim_extractor()
    if extractor is None:
        return []

    claims = extractor.extract_claims(answer)
    results = []

    for claim in claims:
        best_score = -1.0
        best_doc_idx = None
        best_nli_score = 0.0
        best_nli_label = "neutral"
        best_tok = 0.0
        best_ent = 0.0
        best_composite = 0.0
        best_verified = False

        for doc_idx, doc in enumerate(documents):
            result = verify_citation(claim.text, doc)
            score = result["composite_score"]
            nli_label = result.get("nli_label", "neutral")
            verified = result["verified"]

            if score > best_score:
                best_score = score
                best_doc_idx = doc_idx
                best_nli_score = result.get("nli_score", 0.0)
                best_nli_label = nli_label
                best_tok = result["token_overlap_score"]
                best_ent = result["entity_overlap_score"]
                best_composite = score
                best_verified = verified

        results.append(ClaimVerificationResult(
            claim_text=claim.text,
            sentence_idx=claim.sentence_idx,
            claim_idx=claim.claim_idx,
            is_mathematical=claim.is_mathematical,
            is_core_claim=claim.is_core_claim,
            nli_score=best_nli_score,
            nli_label=best_nli_label,
            token_overlap_score=best_tok,
            entity_overlap_score=best_ent,
            composite_score=best_composite,
            verified=best_verified,
            source_doc_idx=best_doc_idx,
            source_doc_page=documents[best_doc_idx].metadata.get("page") if best_doc_idx is not None else None,
        ))

    return results
