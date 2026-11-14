"""CLI entry point for knowledge base ingestion.

Usage:
    python -m src.ingest --source ./docs
    python -m src.ingest --source ./docs/sample.md
"""

import argparse
import sys
from pathlib import Path

from src.retrieval.embedder import FastEmbedEmbedder, MockEmbedder
from src.retrieval.ingest import IngestionPipeline
from src.retrieval.store import DualIndexStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest PDF, Markdown, and TXT documents into Qdrant and BM25 dual stores."
    )
    parser.add_argument(
        "--source",
        "--path",
        dest="source",
        type=str,
        required=True,
        help="Path to a document file (.pdf, .md, .txt) or directory containing documents.",
    )
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default="./data/qdrant_storage",
        help="Storage path for local file-backed Qdrant vectors.",
    )
    parser.add_argument(
        "--bm25-path",
        type=str,
        default="./data/bm25_index.joblib",
        help="Storage path for serialized BM25 index file.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=250,
        help="Chunk window size in tokens (default: 250).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Sliding overlap between chunks in tokens (default: 50).",
    )
    parser.add_argument(
        "--mock-embedder",
        action="store_true",
        help="Use deterministic mock embedder for fast offline testing.",
    )

    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source path does not exist: {source_path}", file=sys.stderr)
        sys.exit(1)

    embedder = MockEmbedder() if args.mock_embedder else FastEmbedEmbedder()
    store = DualIndexStore(
        qdrant_path=args.qdrant_path,
        bm25_path=args.bm25_path,
        embedder=embedder,
    )

    pipeline = IngestionPipeline(
        store=store,
        window_size=args.window_size,
        overlap=args.overlap,
    )

    print(f"Starting ingestion from: {source_path.resolve()}")
    if source_path.is_file():
        result = pipeline.ingest_file(source_path)
    elif source_path.is_dir():
        result = pipeline.ingest_directory(source_path)
    else:
        print(f"Error: Invalid source path type: {source_path}", file=sys.stderr)
        sys.exit(1)

    print("-" * 50)
    print("Ingestion completed successfully.")
    print(f"Status: {result.status}")
    print(f"Files processed: {result.files_processed}")
    print(f"Files skipped (unchanged): {result.files_skipped}")
    print(f"Chunks indexed: {result.chunks_indexed}")
    print(f"Total chunks in store: {store.count()}")
    print("-" * 50)


if __name__ == "__main__":
    main()
