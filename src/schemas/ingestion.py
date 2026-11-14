"""Pydantic schemas for the hybrid knowledge ingestion pipeline."""

from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class IngestionResult(BaseModel):
    """Result of an ingestion run across files or text payloads."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["success", "skipped", "failed", "partial"] = Field(
        ..., description="Overall ingestion outcome: success, skipped, failed, or partial."
    )
    files_processed: int = Field(..., ge=0, description="Number of files processed.")
    files_skipped: int = Field(..., ge=0, description="Number of files skipped (unchanged).")
    chunks_indexed: int = Field(..., ge=0, description="Total number of chunks newly indexed.")
    chunk_ids: List[str] = Field(default_factory=list, description="IDs of newly indexed chunks.")
    source_documents: List[str] = Field(
        default_factory=list, description="Source document names or file paths."
    )


class KnowledgeIngestRequest(BaseModel):
    """Payload schema for POST /knowledge/ingest when submitted as JSON."""

    model_config = ConfigDict(extra="forbid")

    text: Optional[str] = Field(default=None, description="Raw text passage to chunk and index.")
    source_doc: Optional[str] = Field(
        default="api_submission.txt",
        description="Source document identifier for the text passage.",
    )
    source_path: Optional[str] = Field(
        default=None, description="Local file or directory path to ingest from disk."
    )
