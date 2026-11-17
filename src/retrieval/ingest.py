"""Document ingestion orchestration pipeline.

Implements REQ-RET-01, REQ-RET-02, REQ-RET-03, and ADR-0007.
"""

from pathlib import Path
from typing import List, Optional, Union

from src.retrieval.chunker import chunk_text
from src.retrieval.parser import (
    compute_content_hash,
    compute_file_hash,
    extract_text_from_file,
)
from src.retrieval.store import DualIndexStore
from src.schemas.ingestion import IngestionResult


class IngestionPipeline:
    """Orchestrates document parsing, chunking, and dual-index persistence."""

    def __init__(
        self,
        store: Optional[DualIndexStore] = None,
        window_size: int = 250,
        overlap: int = 50,
    ) -> None:
        self._store = store if store is not None else DualIndexStore()
        self._window_size = window_size
        self._overlap = overlap

    @property
    def store(self) -> DualIndexStore:
        return self._store

    def ingest_text(
        self,
        text: str,
        source_doc: str,
        doc_hash: Optional[str] = None,
    ) -> IngestionResult:
        """Ingest raw text directly into the dual index stores."""
        if not doc_hash:
            doc_hash = compute_content_hash(text)

        chunks = chunk_text(
            text,
            source_doc=source_doc,
            window_size=self._window_size,
            overlap=self._overlap,
        )

        added = self._store.add_chunks(chunks, doc_hash=doc_hash)
        if added > 0:
            return IngestionResult(
                status="success",
                files_processed=1,
                files_skipped=0,
                chunks_indexed=added,
                chunk_ids=[c.chunk_id for c in chunks[:added]],
                source_documents=[source_doc],
            )
        else:
            return IngestionResult(
                status="skipped",
                files_processed=0,
                files_skipped=1,
                chunks_indexed=0,
                chunk_ids=[],
                source_documents=[source_doc],
            )

    def ingest_file(self, file_path: Union[Path, str]) -> IngestionResult:
        """Ingest a single PDF, Markdown, or TXT file from disk."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        source_doc = path.name
        doc_hash = compute_file_hash(path)

        if self._store.has_document(source_doc, doc_hash):
            return IngestionResult(
                status="skipped",
                files_processed=0,
                files_skipped=1,
                chunks_indexed=0,
                chunk_ids=[],
                source_documents=[source_doc],
            )

        text = extract_text_from_file(path)
        return self.ingest_text(text, source_doc=source_doc, doc_hash=doc_hash)

    def ingest_directory(
        self,
        dir_path: Union[Path, str],
        recursive: bool = True,
    ) -> IngestionResult:
        """Ingest all supported document files (.pdf, .md, .txt) from a directory."""
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Directory not found: {path}")

        valid_exts = {".pdf", ".md", ".markdown", ".txt"}
        files: List[Path] = [
            p
            for p in (path.rglob("*") if recursive else path.glob("*"))
            if p.is_file() and p.suffix.lower() in valid_exts
        ]

        total_processed = 0
        total_skipped = 0
        total_chunks = 0
        all_chunk_ids: List[str] = []
        source_docs: List[str] = []

        for f in sorted(files):
            res = self.ingest_file(f)
            total_processed += res.files_processed
            total_skipped += res.files_skipped
            total_chunks += res.chunks_indexed
            all_chunk_ids.extend(res.chunk_ids)
            source_docs.extend(res.source_documents)

        return IngestionResult(
            status="success" if (total_chunks > 0 or total_skipped > 0) else "skipped",
            files_processed=total_processed,
            files_skipped=total_skipped,
            chunks_indexed=total_chunks,
            chunk_ids=all_chunk_ids,
            source_documents=source_docs,
        )

    def get_ingestion_telemetry(self) -> dict:
        """Return active knowledge base telemetry and dual-index synchronization statistics."""
        return {
            "window_size": self._window_size,
            "overlap": self._overlap,
            "indexed_chunks_count": len(self._store._chunks),
            "indexed_documents_count": len(self._store._doc_hashes),
            "collection_name": self._store._collection_name,
            "is_in_memory": self._store._in_memory,
        }

