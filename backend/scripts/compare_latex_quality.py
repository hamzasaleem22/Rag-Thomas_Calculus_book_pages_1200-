"""
compare_latex_quality.py
────────────────────────
Compares LaTeX output quality across all available 9-router models.

Queries each model with standard calculus prompts and scores LaTeX output
for spacing artifacts (\\tmspace, \\kern, etc.), frac formatting, and
overall rendering compatibility.
"""

import sys
import time
import json
import re
from openai import OpenAI
from app.config import settings

CALCULUS_PROMPTS = [
    "Find the derivative of f(x) = x^3 sin(x) using the product rule.",
    "Evaluate the integral ∫ x e^x dx using integration by parts.",
    "Use u-substitution to evaluate ∫ sin(x) cos(x) dx.",
    "Apply the chain rule to differentiate h(x) = (x^2 + 1)^5.",
    "Use L'Hôpital's rule to find lim_{x→0} sin(x)/x.",
]

SPACING_PATTERNS = {
    r"\\tmspace": "tmspace (thin space implementation detail)",
    r"\\kern": "kern (explicit kern)",
    r"\\hskip": "hskip (horizontal skip)",
    r"\\vskip": "vskip (vertical skip - shouldn't appear in inline math)",
    r"\\mskip": "mskip (math skip)",
    r"\\mkern": "mkern (math kern)",
    r"\\frac[^\{a-zA-Z\d]": "frac_without_braces",
    r"\\frac\{[^}]*\}[a-zA-Z0-9]\s*[a-zA-Z]": "frac_missing_denom_braces",
    r"\\int[a-z]": "int_missing_space",
    r"\\sum[a-z]": "sum_missing_space",
    r"\\prod[a-z]": "prod_missing_space",
}

BASE_URL = "http://localhost:20128/v1"

MODELS = [
    "kr/claude-sonnet-4.5",
    "kr/deepseek-3.2",
    "kr/minimax-m2.5",
    "cx/gpt-5.5",
    "ag/gemini-3.1-pro-low",
    "kr/qwen3-coder-next",
]


def score_latex_quality(text: str, verbose: bool = False) -> dict:
    """Score a LaTeX answer for quality issues.

    Returns a dict with:
    - total_issues: count of detected spacing/formating issues
    - issues: list of (pattern_name, match_text) tuples
    - has_artifacts: True if any issues found
    """
    issues = []
    for pattern, name in SPACING_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            issues.append((name, m))

    unique_issue_types = set(name for name, _ in issues)
    return {
        "total_issues": len(issues),
        "unique_issue_types": len(unique_issue_types),
        "issue_type_names": sorted(unique_issue_types),
        "issues": issues if verbose else [],
        "has_artifacts": len(issues) > 0,
    }


def query_model(client: OpenAI, model: str, prompt: str) -> str | None:
    """Query a model and return the response text."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a calculus tutor. Answer the user's calculus question "
                        "using proper LaTeX formatting. Use correct braces on \\frac, "
                        "proper subscript/superscript syntax, and valid KaTeX. "
                        "Do NOT use low-level TeX spacing commands like \\tmspace, "
                        "\\kern, \\hskip, \\mskip, or \\mkern. "
                        "Always use \\, for thin spaces and \\; for thick spaces. "
                        "Wrap inline math in $...$ and display math in $$...$$."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
            timeout=60,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[ERROR: {e}]"


def main():
    client = OpenAI(base_url=BASE_URL, api_key=settings.openai_api_key)

    results = {}

    for model in MODELS:
        print(f"\n{'='*70}")
        print(f"  Model: {model}")
        print(f"{'='*70}")

        model_results = []
        total_issues = 0

        for i, prompt in enumerate(CALCULUS_PROMPTS):
            print(f"  Prompt {i+1}/{len(CALCULUS_PROMPTS)}: {prompt[:60]}...", end=" ", flush=True)

            answer = query_model(client, model, prompt)

            if answer is None:
                print("SKIPPED (no response)")
                continue

            if answer.startswith("[ERROR"):
                err_msg = answer[:80]
                print(f"ERROR: {err_msg}")
                continue

            score = score_latex_quality(answer, verbose=False)

            if score["total_issues"] > 0:
                print(f"ISSUES ({score['total_issues']}): {', '.join(score['issue_type_names'])}")
            else:
                print("CLEAN")

            total_issues += score["total_issues"]
            model_results.append({
                "prompt": prompt,
                "score": score,
                "answer_preview": answer[:200],
            })

            time.sleep(1)  # polite delay

        avg_issues = total_issues / len(CALCULUS_PROMPTS) if CALCULUS_PROMPTS else 0
        results[model] = {
            "avg_issues_per_query": avg_issues,
            "total_issues": total_issues,
            "queries_analyzed": len(model_results),
            "details": model_results,
        }

        print(f"\n  >>> {model}: {total_issues} total issues, {avg_issues:.1f} avg per query")

    # Summary table
    print(f"\n{'='*70}")
    print("  SUMMARY: LaTeX Quality by Model")
    print(f"{'='*70}")
    print(f"  {'Model':<30} {'Avg Issues':<15} {'Total':<10}")
    print(f"  {'-'*30} {'-'*15} {'-'*10}")
    for model, r in sorted(results.items(), key=lambda x: x[1]["avg_issues_per_query"]):
        print(f"  {model:<30} {r['avg_issues_per_query']:<15.1f} {r['total_issues']:<10}")
    print(f"{'='*70}")

    # Write report to file
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models_tested": MODELS,
        "prompts": CALCULUS_PROMPTS,
        "results": {k: {
            "avg_issues_per_query": v["avg_issues_per_query"],
            "total_issues": v["total_issues"],
            "queries_analyzed": v["queries_analyzed"],
        } for k, v in results.items()},
        "ranking": sorted(results.items(), key=lambda x: x[1]["avg_issues_per_query"]),
    }

    report_path = "scripts/latex_quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {report_path}")

    return results


if __name__ == "__main__":
    results = main()
    sys.exit(0)
