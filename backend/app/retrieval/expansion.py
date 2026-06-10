import re
import math

MATH_SYNONYMS = {
    "l'hôpital": ["l'hopital", "l'hospital", "lhopital"],
    "l'hopital": ["l'hôpital", "l'hospital", "lhopital"],
    "l'hospital": ["l'hôpital", "l'hopital", "lhopital"],
    "derivative": ["differentiation", "differential"],
    "integral": ["integration", "antiderivative"],
    "theorem": ["law", "principle", "rule"],
    "gradient": ["nabla", "del"],
    "divergence": ["gauss", "gauss's theorem", "gauss theorem"],
    "divergence theorem": ["gauss theorem", "gauss's theorem", "flux integral theorem"],
    "flux": ["surface integral", "outward flux", "net flux"],
    "gauss law": ["gauss's law", "electric flux"],
    "curl": ["rotation", "rot"],
    "riemann sum": ["riemann summation", "sigma sum"],
    "taylor series": ["taylor expansion", "maclaurin series"],
    "chain rule": ["function composition", "composite function"],
    "product rule": ["leibniz rule"],
    "fundamental theorem of calculus": ["ftc", "fundamental theorem"],
    "partial derivative": ["partial differentiation", "partial"],
    "double integral": ["iterated integral", "multiple integral"],
    "vector field": ["vector valued function"],
    "laplace transform": ["laplace"],
    "fourier series": ["fourier expansion"],
    "eigenvalue": ["characteristic value"],
    "eigenvector": ["characteristic vector"],
    "limit": ["approaches", "tends to"],
    "continuous": ["continuity", "unbroken"],
    "converge": ["convergent", "convergence", "limit exists"],
    "diverge": ["divergent", "divergence"],
    "function": ["mapping", "map"],
    "sequence": ["series", "progression"],
    "area": ["surface area", "region"],
    "volume": ["solid", "cross-section"],
    "slope": ["tangent", "gradient", "rate of change"],
    "velocity": ["speed", "rate of change of position"],
    "acceleration": ["second derivative of position"],
}

SYNONYM_PATTERNS = [
    (re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE), v)
    for k, v in MATH_SYNONYMS.items()
]

MULTI_HOP_CONNECTORS = re.compile(r"\b(and|then|also|plus|combined with|together with)\b", re.IGNORECASE)
MULTI_HOP_INDICATORS = [
    r'\b(?:compare|contrast|differentiate between)\b',
    r'\b(?:first|then|after|before|subsequently)\b',
    r'\b(?:both|relationship between)\b',
    r'\b(?:how does|why does|what happens when)\b',
    r'(?:using.*?find|given.*?compute)',
    r'(?:derive|determine).*(?:and|then).*(?:using|from)',
]


def detect_query_type(query: str) -> str:
    query_lower = query.lower().strip()

    formula_indicators = [
        r"\\frac", r"\\int", r"\\sum", r"\\lim", r"\\partial",
        r"d/dx", r"f\(", r"y\s*=", r"f'\(", r"\\infty",
        r"\$\$", r"\$",
    ]
    for indicator in formula_indicators:
        if re.search(indicator, query):
            return "equation"

    theorem_indicators = [
        r"\btheorem\b", r"\bproof\b", r"\bprove\b", r"\bstate\b",
        r"\bdefinition\b", r"\brule\b",
    ]
    for indicator in theorem_indicators:
        if re.search(indicator, query_lower):
            return "theorem"

    definition_indicators = [
        r"\bwhat is\b", r"\bdefine\b", r"\bexplain\b", r"\bmean\b",
        r"\bwhat does\b",
    ]
    for indicator in definition_indicators:
        if re.search(indicator, query_lower):
            return "definition"

    return "general"


def detect_chapter_reference(query: str) -> list[int]:
    chapter_refs = re.findall(r"(?:Chapter|Ch\.?)\s*(\d+)", query, re.IGNORECASE)
    section_refs = re.findall(r"(?:Section|Sec\.?)\s*(\d+)\.(\d+)", query, re.IGNORECASE)
    result = set()
    for ch in chapter_refs:
        result.add(int(ch))
    for ch, _ in section_refs:
        result.add(int(ch))
    return sorted(result)


def is_multi_hop_query(query: str) -> bool:
    query_lower = query.lower()
    if MULTI_HOP_CONNECTORS.search(query) and len(query.split()) > 8:
        return True
    for pattern in MULTI_HOP_INDICATORS:
        if re.search(pattern, query_lower):
            return True
    return False


def decompose_multi_hop(query: str) -> list[str]:
    if not is_multi_hop_query(query):
        return [query]

    # Try connector-based split
    if MULTI_HOP_CONNECTORS.search(query) and len(query.split()) > 8:
        parts = MULTI_HOP_CONNECTORS.split(query)
        sub_queries = [p.strip() for p in parts if p.strip() and not MULTI_HOP_CONNECTORS.match(p.strip())]
        if len(sub_queries) >= 2:
            return sub_queries + [query]

    # Try LLM-based decomposition for complex queries
    try:
        from app.config import settings
        from openai import OpenAI
        client = OpenAI(
            api_key=settings.openai_api_key or "no-key",
            base_url=settings.openai_base_url,
        )
        decompose_prompt = (
            "Decompose the following math question into 2-3 simpler sub-questions. "
            "Return one sub-question per line, numbered. Only respond with the sub-questions:\n\n"
            f"Question: {query}"
        )
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": decompose_prompt}],
            temperature=0.1,
            max_tokens=256,
        )
        text = response.choices[0].message.content.strip()
        sub_queries = [
            re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            for line in text.split('\n')
            if line.strip() and not line.strip().startswith('Question')
        ]
        if len(sub_queries) >= 2:
            return sub_queries + [query]
    except Exception as e:
        print(f"  LLM decomposition failed: {e}")

    return [query]


def expand_query(query: str) -> list[str]:
    queries = [query]
    for pattern, synonyms in SYNONYM_PATTERNS:
        if pattern.search(query):
            for syn in synonyms:
                expanded = pattern.sub(syn, query)
                if expanded != query and expanded not in queries:
                    queries.append(expanded)
    return queries


def expand_query_text(query: str) -> str:
    expanded_queries = expand_query(query)
    return " ".join(expanded_queries)
