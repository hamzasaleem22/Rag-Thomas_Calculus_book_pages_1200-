import re

MATH_SYNONYMS = {
    "l'hôpital": ["l'hopital", "l'hospital", "lhopital"],
    "l'hopital": ["l'hôpital", "l'hospital", "lhopital"],
    "l'hospital": ["l'hôpital", "l'hopital", "lhopital"],
    "derivative": ["differentiation", "differential"],
    "integral": ["integration", "antiderivative"],
    "theorem": ["law", "principle", "rule"],
    "gradient": ["nabla", "del"],
    "divergence": ["gauss", "gauss's theorem", "gauss theorem"],
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
}

SYNONYM_PATTERNS = [
    (re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE), v)
    for k, v in MATH_SYNONYMS.items()
]


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
