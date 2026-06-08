import json, sys, os, math
os.environ["OPENAI_API_KEY"] = "sk-71ac1c63eb0ace5c-hih7bl-c3910a8f"
os.environ["OPENAI_BASE_URL"] = "http://localhost:20128/v1"
os.environ["OPENAI_API_BASE"] = "http://localhost:20128/v1"

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from app.config import settings
from app.retrieval.vector_store import HybridVectorStore
from app.retrieval.reranker import Reranker
from app.generation.generator import Generator
import pickle

with open("/tmp/qdrant_calculus.pkl", "rb") as f:
    client = pickle.load(f)

store = HybridVectorStore(client=client)
reranker = Reranker()
generator = Generator()

questions = [
    {
        "question": "What is the derivative of sin(x)?",
        "ground_truth": "The derivative of sin(x) is cos(x).",
    },
    {
        "question": "State the Fundamental Theorem of Calculus, Part 1.",
        "ground_truth": "If f is continuous on [a,b] and F is an antiderivative of f, then ∫_a^b f(x) dx = F(b) - F(a).",
    },
    {
        "question": "What is the formula for the area of a circle?",
        "ground_truth": "The area of a circle is A = πr², where r is the radius.",
    },
    {
        "question": "What is the chain rule for differentiation?",
        "ground_truth": "If y = f(u) and u = g(x), then dy/dx = dy/du · du/dx = f'(g(x)) · g'(x).",
    },
    {
        "question": "What is L'Hôpital's rule?",
        "ground_truth": "If f and g are differentiable and lim f(x)/g(x) is indeterminate (0/0 or ∞/∞), then lim f(x)/g(x) = lim f'(x)/g'(x), provided the latter limit exists.",
    },
]

answers = []
contexts = []
for q in questions:
    print(f"Processing: {q['question'][:50]}...")
    docs = store.similarity_search(q["question"], k=10)

    texts = [d.page_content for d in docs]
    reranked = reranker.rerank(q["question"], texts, top_k=5)
    kept = {t for t, _ in reranked}
    docs = [d for d in docs if d.page_content in kept]

    result = generator.generate(q["question"], docs)
    answers.append(result["answer"])
    contexts.append([[d.page_content for d in docs]])

    print(f"  Answer length: {len(result['answer'])} chars, {len(result['citations'])} citations")

data = {
    "question": [q["question"] for q in questions],
    "answer": answers,
    "contexts": [c[0] for c in contexts],
    "ground_truth": [q["ground_truth"] for q in questions],
}

dataset = Dataset.from_dict(data)

print("\nEvaluating with RAGAS...")
from langchain_openai import ChatOpenAI
eval_llm = ChatOpenAI(
    model=settings.llm_model,
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
    temperature=0,
)
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=eval_llm,
)

print("\n" + "=" * 50)
print("RAGAS Evaluation Results")
print("=" * 50)
try:
    scores = dict(result)
    for k, v in scores.items():
        if isinstance(v, float) and not math.isnan(v):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: N/A")
except Exception:
    print(f"Result: {result}")
    scores = {}

with open("data/ragas_results.json", "w") as f:
    json.dump(scores, f, indent=2, default=str)
print("\nSaved to data/ragas_results.json")
