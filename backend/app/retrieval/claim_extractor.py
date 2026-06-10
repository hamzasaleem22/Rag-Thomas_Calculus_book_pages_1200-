import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Claim:
    text: str
    sentence_idx: int
    claim_idx: int
    is_mathematical: bool = False
    has_citation: bool = False
    citation_indices: list[int] = field(default_factory=list)
    is_core_claim: bool = False


class AtomicClaimExtractor:
    COORDINATION_PATTERNS = [
        r'\s+,\s+(?:and|or|while|whereas|but)\s+',
        r'\s+and\s+(?=(?:the|this|its|their|we|it|a|an|every|each|all|any|no))',
        r'\s+;\s+',
    ]
    NON_SPLIT_PATTERNS = [
        r'\$[^$]*\$',
        r'\$\$[^$]*\$\$',
        r'\\textbf\{[^}]*\}',
        r'\\frac\{[^}]*\}\{[^}]*\}',
        r'\\sum\{[^}]*\}',
        r'\\int\{[^}]*\}',
        r'f\(x\)\s*=\s*.+?\s+and\s+g\(x\)\s*=',
        r'both\s+.+?\s+and\s+.+?',
    ]

    def extract_claims(self, answer: str) -> list[Claim]:
        sentences = self._split_sentences(answer)
        claims = []
        for sent_idx, sentence in enumerate(sentences):
            sent_claims = self._decompose_sentence(sentence, sent_idx)
            claims.extend(sent_claims)
        return claims

    def _split_sentences(self, text: str) -> list[str]:
        placeholders = {}
        def _protect(m):
            key = f"__LATEX_{len(placeholders)}__"
            placeholders[key] = m.group(0)
            return key
        text_protected = re.sub(r'\$\$.*?\$\$', _protect, text, flags=re.DOTALL)
        text_protected = re.sub(r'\$[^$]*?\$', _protect, text_protected)
        text_protected = re.sub(r'\\\[.*?\\\]', _protect, text_protected, flags=re.DOTALL)
        text_protected = re.sub(r'\\\(.*?\\\)', _protect, text_protected, flags=re.DOTALL)

        raw = re.split(r'(?<=[.!?])\s+', text_protected)
        sentences = []
        for s in raw:
            for key, val in placeholders.items():
                s = s.replace(key, val)
            s = s.strip()
            if s:
                sentences.append(s)
        return sentences

    def _decompose_sentence(self, sentence: str, sent_idx: int) -> list[Claim]:
        for pattern in self.NON_SPLIT_PATTERNS:
            if re.search(pattern, sentence):
                return [self._make_claim(sentence, sent_idx, 0)]

        citations = re.findall(r'\[(\d+(?:,\s*\d+)*)\]', sentence)
        has_cite = len(citations) > 0
        cite_indices = []
        if citations:
            for c in citations:
                cite_indices.extend(int(x) for x in re.findall(r'\d+', c))

        parts = self._smart_split(sentence)
        if len(parts) <= 1:
            return [self._make_claim(sentence, sent_idx, 0, has_cite, cite_indices)]

        claims = []
        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                claims.append(self._make_claim(part, sent_idx, i, has_cite, cite_indices))
        return claims if claims else [self._make_claim(sentence, sent_idx, 0, has_cite, cite_indices)]

    def _smart_split(self, sentence: str) -> list[str]:
        protected = sentence
        for m in re.finditer(r'\$[^$]*\$', sentence):
            protected = protected.replace(m.group(0), m.group(0).replace(' and ', ' __AND__ ').replace(' or ', ' __OR__ '))
        for m in re.finditer(r'\$\$[^$]*\$\$', sentence):
            protected = protected.replace(m.group(0), m.group(0).replace(' and ', ' __AND__ ').replace(' or ', ' __OR__ '), 1)

        for pattern in self.COORDINATION_PATTERNS:
            parts = re.split(pattern, protected)
            if len(parts) > 1:
                parts = [p.replace(' __AND__ ', ' and ').replace(' __OR__ ', ' or ').strip() for p in parts]
                return parts

        parts = [protected.replace(' __AND__ ', ' and ').replace(' __OR__ ', ' or ').strip()]
        return parts

    def _make_claim(self, text: str, sent_idx: int, claim_idx: int,
                    has_cite: bool = False, cite_indices: list[int] | None = None) -> Claim:
        # Strip citation markers from text
        clean_text = re.sub(r'\s*\[\d+(?:,\s*\d+)*\]\s*', '', text).strip()
        clean_text = re.sub(r'\s+', ' ', clean_text)
        # Fix trailing space before period
        clean_text = re.sub(r'\s+\.', '.', clean_text)
        is_math = bool(re.search(r'[\$\\]', clean_text))
        core_keywords = ['theorem', 'definition', 'rule', 'formula', 'property',
                         'derivative', 'integral', 'limit', 'function', 'equation',
                         'continuous', 'differentiable']
        is_core = is_math or any(kw in clean_text.lower() for kw in core_keywords)
        return Claim(
            text=clean_text,
            sentence_idx=sent_idx,
            claim_idx=claim_idx,
            is_mathematical=is_math,
            has_citation=has_cite,
            citation_indices=list(set(cite_indices)) if cite_indices else [],
            is_core_claim=is_core,
        )
