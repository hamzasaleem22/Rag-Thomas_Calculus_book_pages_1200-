#!/usr/bin/env python3
"""Evaluate chunk quality: metadata completeness, size distribution, boundary integrity."""
import json, sys, re
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

MIN_CHUNK_CHARS = 100
MAX_CHUNK_CHARS = 3100
REQUIRED_META_FIELDS = ["page", "chapter", "section", "content_type", "chapter_number"]
OPTIONAL_META_FIELDS = ["section_number", "equation_density", "has_formula", "key_terms", "chunk_size_tokens"]
LATEX_DELIMITERS = re.compile(r"\$\$|\$|\\\[|\\\]|\\\(|\\\)")


def load_chunks(path: str = "data/markdown/parsed_docs.jsonl") -> list[dict]:
    chunks = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            chunks.append({
                "page_content": data.get("page_content", ""),
                "metadata": data.get("metadata", {}),
            })
    return chunks


def evaluate_chunk_sizes(chunks: list[dict]) -> dict:
    sizes = [len(c["page_content"]) for c in chunks]
    within_range = sum(1 for s in sizes if MIN_CHUNK_CHARS <= s <= MAX_CHUNK_CHARS)
    return {
        "total_chunks": len(chunks),
        "min_chars": min(sizes) if sizes else 0,
        "max_chars": max(sizes) if sizes else 0,
        "avg_chars": round(sum(sizes) / max(len(sizes), 1)),
        "within_range_pct": round(within_range / max(len(chunks), 1), 3),
        "compliant": within_range == len(chunks),
    }


def evaluate_metadata_completeness(chunks: list[dict]) -> dict:
    results = {}
    for field in REQUIRED_META_FIELDS:
        present = sum(1 for c in chunks if c["metadata"].get(field) is not None and c["metadata"].get(field) != "")
        results[field] = round(present / max(len(chunks), 1), 3)

    all_required_present = all(v >= 0.90 for v in results.values())
    return {
        "field_coverage": results,
        "all_required_above_90pct": all_required_present,
    }


def evaluate_latex_integrity(chunks: list[dict]) -> dict:
    import re
    currency = re.compile(r"\$\s*\d[\d,.\s]*")
    garbled_dollar = re.compile(r"[a-zA-Z]\$")
    isolated_dollar = re.compile(r"(?:^|\s)\$(?:$|\s)")
    broken = 0
    for c in chunks:
        text = currency.sub("", c["page_content"])
        text = garbled_dollar.sub("", text)
        text = isolated_dollar.sub("", text)
        singles = text.count("$") % 2
        doubles = text.count("$$") % 2
        brackets = (text.count("\\[") - text.count("\\]"))
        parens = (text.count("\\(") - text.count("\\)"))
        if singles or doubles or brackets or parens:
            broken += 1
    return {
        "broken_chunks": broken,
        "broken_pct": round(broken / max(len(chunks), 1), 3),
        "intact_pct": round(1 - broken / max(len(chunks), 1), 3),
        "compliant": broken == 0,
    }


def evaluate_content_type_distribution(chunks: list[dict]) -> dict:
    types = Counter(c["metadata"].get("content_type", "unknown") for c in chunks)
    return {
        "distribution": dict(types),
        "unique_types": len(types),
    }


def evaluate_chapter_coverage(chunks: list[dict]) -> dict:
    chapters = Counter()
    for c in chunks:
        ch = c["metadata"].get("chapter", "")
        ch_num = c["metadata"].get("chapter_number", 0)
        if ch_num > 0:
            chapters[ch_num] = chapters.get(ch_num, 0) + 1
    missing = [ch for ch in range(1, 17) if ch not in chapters]
    return {
        "chapters_covered": len(chapters),
        "chapters_missing": missing,
        "coverage_pct": round(len(chapters) / 16, 3),
        "compliant": len(missing) == 0,
    }


def run_evaluation():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks\n")

    print("=" * 60)
    print("CHUNK QUALITY EVALUATION")
    print("=" * 60)

    size_results = evaluate_chunk_sizes(chunks)
    print(f"\n📏 Size Distribution:")
    print(f"  Total chunks: {size_results['total_chunks']}")
    print(f"  Min: {size_results['min_chars']} chars")
    print(f"  Max: {size_results['max_chars']} chars")
    print(f"  Avg: {size_results['avg_chars']} chars")
    print(f"  Within range [{MIN_CHUNK_CHARS}-{MAX_CHUNK_CHARS}]: {size_results['within_range_pct']*100:.1f}%")
    print(f"  ✅ PASS" if size_results["compliant"] else f"  ❌ FAIL")

    meta_results = evaluate_metadata_completeness(chunks)
    print(f"\n📋 Metadata Completeness:")
    for field, pct in meta_results["field_coverage"].items():
        status = "✅" if pct >= 0.90 else "❌"
        print(f"  {status} {field}: {pct*100:.1f}%")
    print(f"  {'✅ PASS' if meta_results['all_required_above_90pct'] else '❌ FAIL'}")

    latex_results = evaluate_latex_integrity(chunks)
    print(f"\n🔬 LaTeX Integrity:")
    print(f"  Broken chunks: {latex_results['broken_chunks']}")
    print(f"  Broken: {latex_results['broken_pct']*100:.1f}%")
    print(f"  Intact: {latex_results['intact_pct']*100:.1f}%")
    print(f"  {'✅ PASS' if latex_results['compliant'] else '❌ FAIL'}")

    type_results = evaluate_content_type_distribution(chunks)
    print(f"\n📊 Content Type Distribution:")
    for ct, count in sorted(type_results["distribution"].items(), key=lambda x: -x[1]):
        print(f"  {ct}: {count}")

    chapter_results = evaluate_chapter_coverage(chunks)
    print(f"\n📚 Chapter Coverage:")
    print(f"  Covered: {chapter_results['chapters_covered']}/16")
    if chapter_results["chapters_missing"]:
        print(f"  Missing: {chapter_results['chapters_missing']}")
    print(f"  Coverage: {chapter_results['coverage_pct']*100:.1f}%")
    print(f"  {'✅ PASS' if chapter_results['compliant'] else '❌ FAIL'}")

    # Overall score
    total = 4
    passed = sum([
        size_results["compliant"],
        meta_results["all_required_above_90pct"],
        latex_results["compliant"],
        chapter_results["compliant"],
    ])
    print(f"\n{'=' * 60}")
    print(f"OVERALL: {passed}/{total} checks passed")
    print(f"{'=' * 60}")
    return passed == total


if __name__ == "__main__":
    success = run_evaluation()
    sys.exit(0 if success else 1)
