#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient

from app.ingestion.cleaner import clean_documents
from app.ingestion.indexer import index_documents
from app.ingestion.pdf_loader import MarkerDocumentLoader
from app.ingestion.chunker import chunk_documents


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF into RAG pipeline")
    parser.add_argument("pdf_path", nargs="?", default="Thomas' Calculus Early Transcendentals, 14th Edition.pdf",
                        help="Path to the PDF file")
    parser.add_argument("--output-dir", default="data/markdown",
                        help="Output directory for markdown and parsed JSONL")
    parser.add_argument("--save-chunks", action="store_true",
                        help="Save chunks to JSONL")
    parser.add_argument("--skip-index", action="store_true",
                        help="Skip Qdrant indexing")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    output_dir = args.output_dir

    print(f"Converting {pdf_path} to markdown...")
    loader = MarkerDocumentLoader(pdf_path, output_dir)
    docs = loader.load()
    print(f"Loaded {len(docs)} page documents")

    print("Cleaning documents...")
    docs = clean_documents(docs)
    print(f"Cleaned {len(docs)} documents")

    print("Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    if args.save_chunks:
        output_path = Path(output_dir) / "parsed_docs.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for chunk in chunks:
                f.write(chunk.model_dump_json() + "\n")
        print(f"Saved {len(chunks)} chunks to {output_path}")

    if not args.skip_index:
        print("Indexing into Qdrant (in-memory)...")
        client = QdrantClient(":memory:")
        count = index_documents(chunks, client=client)
        print(f"Indexed {count} vectors into Qdrant collection")
    else:
        print("Skipped Qdrant indexing")


if __name__ == "__main__":
    main()
