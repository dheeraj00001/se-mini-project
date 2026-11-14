"""Unit tests for Hybrid Knowledge Ingestion & Deterministic Dual-Index Store.

Covers RTM Traceability:
- UT_RET_01: PDF/MD/TXT parsing and 250-300 token chunking with 50-token overlap (REQ-RET-01).
- UT_RET_02: SHA-256 deterministic chunk ID calculation & metadata consistency (REQ-RET-02).
- UT_RET_03: Persistent Qdrant & BM25 store synchronization, restart reload,
  and duplicate skipping (REQ-RET-03).
"""

import io
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from src.gateway.app import create_app
from src.retrieval.chunker import chunk_text
from src.retrieval.embedder import (
    EmbeddingGemmaEmbedder,
    FastEmbedEmbedder,
    MockEmbedder,
    get_default_embedder,
)
from src.retrieval.ingest import IngestionPipeline
from src.retrieval.parser import (
    compute_content_hash,
    compute_file_hash,
    extract_text_from_file,
)
from src.retrieval.store import DualIndexStore
from src.schemas.verification import EvidenceChunk

# ============================================================================
# Helpers & Fixtures
# ============================================================================


def create_dummy_pdf(path: Path, pages_text: List[str]) -> Path:
    """Helper to generate a real PDF file with the specified text pages."""
    writer = PdfWriter()
    for text in pages_text:
        # pypdf 4+ allows adding a blank page and annotating or writing text
        # Or creating a simple page with text:
        writer.add_blank_page(width=72 * 8.5, height=72 * 11)
    # Using pypdf's low-level stream or writing standard text:
    # To keep it robust without extra heavy fonts:
    with open(path, "wb") as f:
        writer.write(f)

    # Let's write text stream directly if pypdf blank page doesn't have text stream
    # Instead, we can write a valid minimal PDF with extractable text:
    content = ""
    for i, t in enumerate(pages_text):
        content += f"BT /F1 12 Tf 72 {700 - i*20} Td ({t}) Tj ET\n"

    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        b"4 0 obj << /Length "
        + str(len(content)).encode("ascii")
        + b" >> stream\n"
        + content.encode("latin-1")
        + b"\nendstream\nendobj\n"
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"xref\n0 6\n0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000262 00000 n \n"
        b"0000000370 00000 n \n"
        b"trailer << /Size 6 /Root 1 0 R >>\n"
        b"startxref\n450\n%%EOF\n"
    )
    path.write_bytes(pdf_bytes)
    return path


@pytest.fixture
def temp_store_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory for dual stores."""
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def mock_store(temp_store_dir: Path) -> DualIndexStore:
    """Fixture providing a file-backed DualIndexStore with a fast MockEmbedder."""
    qdrant_path = str(temp_store_dir / "qdrant_storage")
    bm25_path = str(temp_store_dir / "bm25_index.joblib")
    return DualIndexStore(
        qdrant_path=qdrant_path,
        bm25_path=bm25_path,
        collection_name="test_kb",
        embedder=MockEmbedder(dimension=384),
        in_memory=False,
    )


# ============================================================================
# UT_RET_01: Document Parsing & Token Window Chunking (REQ-RET-01)
# ============================================================================


def test_ut_ret_01_parse_markdown_file(tmp_path: Path) -> None:
    """Verify parsing and extracting text from a Markdown (.md) document."""
    md_file = tmp_path / "sample.md"
    content = "# Architectural Overview\n\nDehallucination middleware intercepts LLM responses."
    md_file.write_text(content, encoding="utf-8")

    extracted = extract_text_from_file(md_file)
    assert "Architectural Overview" in extracted
    assert "Dehallucination middleware intercepts LLM responses." in extracted


def test_ut_ret_01_parse_txt_file(tmp_path: Path) -> None:
    """Verify parsing plain text (.txt) files."""
    txt_file = tmp_path / "notes.txt"
    content = "The NVIDIA GeForce RTX 2050 has 4GB VRAM and dedicated CUDA cores."
    txt_file.write_text(content, encoding="utf-8")

    extracted = extract_text_from_file(txt_file)
    assert extracted.strip() == content


def test_ut_ret_01_parse_pdf_file(tmp_path: Path) -> None:
    """Verify extracting text from PDF files using pypdf."""
    pdf_file = tmp_path / "test_doc.pdf"
    create_dummy_pdf(pdf_file, ["Autonomous Grounding Verification"])

    extracted = extract_text_from_file(pdf_file)
    assert "Autonomous Grounding Verification" in extracted


def test_ut_ret_01_unsupported_file_extension(tmp_path: Path) -> None:
    """Verify rejecting unsupported document formats with ValueError."""
    unsupported_file = tmp_path / "table.csv"
    unsupported_file.write_text("col1,col2\nval1,val2", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported document format"):
        extract_text_from_file(unsupported_file)


def test_ut_ret_01_nonexistent_file(tmp_path: Path) -> None:
    """Verify FileNotFoundError on missing document path."""
    missing = tmp_path / "nonexistent.md"
    with pytest.raises(FileNotFoundError):
        extract_text_from_file(missing)


def test_ut_ret_01_chunking_empty_or_whitespace_text() -> None:
    """Verify chunker returns empty list for blank or empty text."""
    assert chunk_text("", source_doc="empty.txt") == []
    assert chunk_text("   \n\t  ", source_doc="blank.txt") == []


def test_ut_ret_01_chunking_short_text_single_chunk() -> None:
    """Verify text shorter than window_size produces exactly one chunk."""
    short_text = "This is a short text passage with only ten words total."
    chunks = chunk_text(short_text, source_doc="short.txt", window_size=250, overlap=50)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source_doc == "short.txt"
    assert chunk.tier == 2
    assert chunk.text == short_text
    assert chunk.metadata["chunk_index"] == 0
    assert chunk.metadata["token_count"] == 11


def test_ut_ret_01_chunking_sliding_window_with_overlap() -> None:
    """Verify 250-token window with 50-token overlap creates correct spans."""
    # Generate 550 words: "word_0 word_1 ... word_549"
    words = [f"token_{i}" for i in range(550)]
    full_text = " ".join(words)

    # Window=250, Overlap=50 -> Step = 200
    # Chunk 0: tokens 0..250 (250 tokens)
    # Chunk 1: tokens 200..450 (250 tokens)
    # Chunk 2: tokens 400..550 (150 tokens)
    chunks = chunk_text(full_text, source_doc="corpus.txt", window_size=250, overlap=50)

    assert len(chunks) == 3

    assert chunks[0].metadata["start_token"] == 0
    assert chunks[0].metadata["end_token"] == 250
    assert chunks[0].metadata["token_count"] == 250
    assert chunks[0].text.startswith("token_0 ")
    assert chunks[0].text.endswith("token_249")

    assert chunks[1].metadata["start_token"] == 200
    assert chunks[1].metadata["end_token"] == 450
    assert chunks[1].metadata["token_count"] == 250
    assert chunks[1].text.startswith("token_200 ")
    assert chunks[1].text.endswith("token_449")

    # Overlap between chunk 0 and chunk 1 is tokens 200..249
    overlap_words = words[200:250]
    for w in overlap_words:
        assert w in chunks[0].text
        assert w in chunks[1].text

    assert chunks[2].metadata["start_token"] == 400
    assert chunks[2].metadata["end_token"] == 550
    assert chunks[2].metadata["token_count"] == 150
    assert chunks[2].text.startswith("token_400 ")
    assert chunks[2].text.endswith("token_549")


def test_ut_ret_01_invalid_chunking_parameters() -> None:
    """Verify ValueError is raised if overlap >= window_size or window_size <= 0."""
    with pytest.raises(ValueError, match="window_size must be greater than overlap"):
        chunk_text("Some text", source_doc="doc.txt", window_size=50, overlap=50)

    with pytest.raises(ValueError, match="window_size must be greater than 0"):
        chunk_text("Some text", source_doc="doc.txt", window_size=0, overlap=0)


# ============================================================================
# UT_RET_02: SHA-256 Deterministic Chunk ID Hashing (REQ-RET-02)
# ============================================================================


def test_ut_ret_02_deterministic_hash_consistency() -> None:
    """Verify identical text and metadata produce identical SHA-256 chunk IDs."""
    text = "The DeBERTa-v3 cross encoder executes on CPU threads."
    chunks_1 = chunk_text(text, source_doc="spec.md", window_size=250, overlap=50)
    chunks_2 = chunk_text(text, source_doc="spec.md", window_size=250, overlap=50)

    assert len(chunks_1) == 1
    assert len(chunks_2) == 1
    assert chunks_1[0].chunk_id == chunks_2[0].chunk_id
    # SHA-256 hex string has 64 characters
    assert len(chunks_1[0].chunk_id) == 64
    assert all(c in "0123456789abcdef" for c in chunks_1[0].chunk_id)


def test_ut_ret_02_tamper_sensitivity_and_source_uniqueness() -> None:
    """Verify hash changes if text or source_doc changes."""
    text_a = "Gemma 4B is quantized to 4-bit precision."
    text_b = "Gemma 4B is quantized to 8-bit precision."

    chunk_a = chunk_text(text_a, source_doc="hardware.md")[0]
    chunk_b = chunk_text(text_b, source_doc="hardware.md")[0]
    chunk_diff_source = chunk_text(text_a, source_doc="different.md")[0]

    assert chunk_a.chunk_id != chunk_b.chunk_id
    assert chunk_a.chunk_id != chunk_diff_source.chunk_id


def test_ut_ret_02_evidence_chunk_contract_fidelity() -> None:
    """Verify chunk output conforms strictly to EvidenceChunk Pydantic schema."""
    text = "Reciprocal Rank Fusion merges dense and sparse scores."
    chunks = chunk_text(text, source_doc="rrf.txt", tier=2)
    chunk = chunks[0]

    assert isinstance(chunk, EvidenceChunk)
    assert chunk.tier == 2
    assert chunk.source_doc == "rrf.txt"
    assert chunk.score is None
    assert "chunk_index" in chunk.metadata
    assert "token_count" in chunk.metadata


def test_ut_ret_02_compute_file_and_content_hash(tmp_path: Path) -> None:
    """Verify SHA-256 calculation for string content and disk files."""
    content = "Deterministic hash verification."
    c_hash = compute_content_hash(content)
    assert len(c_hash) == 64

    f = tmp_path / "hash_target.txt"
    f.write_text(content, encoding="utf-8")
    f_hash = compute_file_hash(f)
    assert f_hash == c_hash

    with pytest.raises(FileNotFoundError):
        compute_file_hash(tmp_path / "missing_file.txt")


# ============================================================================
# UT_RET_03: Persistent Dual-Index Store Sync & Duplicate Skipping (REQ-RET-03)
# ============================================================================


def test_ut_ret_03_store_synchronization_and_chunk_retrieval(mock_store: DualIndexStore) -> None:
    """Verify chunks are added to both Qdrant and BM25 with shared chunk IDs."""
    text = "Qdrant handles dense vector similarity while BM25 handles sparse keyword search."
    chunks = chunk_text(text, source_doc="dual_store.md")

    added = mock_store.add_chunks(chunks, doc_hash="hash_123")
    assert added == len(chunks)
    assert mock_store.count() == len(chunks)

    # Check retrieval from memory/index
    retrieved = mock_store.get_chunk(chunks[0].chunk_id)
    assert retrieved is not None
    assert retrieved.chunk_id == chunks[0].chunk_id
    assert retrieved.text == chunks[0].text
    assert retrieved.source_doc == "dual_store.md"


def test_ut_ret_03_persistence_across_store_restarts(temp_store_dir: Path) -> None:
    """Verify BM25 index and Qdrant persistence survive store reload (restart)."""
    qdrant_path = str(temp_store_dir / "qdrant_storage")
    bm25_path = str(temp_store_dir / "bm25_index.joblib")

    # Instance 1: Ingest knowledge
    store1 = DualIndexStore(
        qdrant_path=qdrant_path,
        bm25_path=bm25_path,
        collection_name="restart_test",
        embedder=MockEmbedder(),
    )
    text = "Persistent storage guarantees durability across application restarts."
    chunks = chunk_text(text, source_doc="durability.txt")
    store1.add_chunks(chunks, doc_hash="doc_hash_restart")
    assert store1.count() == 1

    # Close store 1 (release lock if applicable)
    store1.close()

    # Instance 2: Reload from disk files
    store2 = DualIndexStore(
        qdrant_path=qdrant_path,
        bm25_path=bm25_path,
        collection_name="restart_test",
        embedder=MockEmbedder(),
    )
    assert store2.count() == 1
    reloaded_chunk = store2.get_chunk(chunks[0].chunk_id)
    assert reloaded_chunk is not None
    assert reloaded_chunk.text == text
    assert store2.has_document("durability.txt", "doc_hash_restart") is True
    store2.close()


def test_ut_ret_03_reingestion_skips_unchanged_documents(mock_store: DualIndexStore) -> None:
    """Verify unchanged files with matching hash skip re-computation and re-indexing."""
    text = "Deterministic deduplication avoids re-indexing unchanged documents."
    chunks = chunk_text(text, source_doc="dedup.md")
    doc_hash = "sha256_constant_hash"

    # First ingestion
    added_1 = mock_store.add_chunks(chunks, doc_hash=doc_hash)
    assert added_1 == 1
    assert mock_store.count() == 1

    # Second ingestion of unchanged document
    added_2 = mock_store.add_chunks(chunks, doc_hash=doc_hash)
    assert added_2 == 0
    assert mock_store.count() == 1


def test_ut_ret_03_ingestion_pipeline_directory(tmp_path: Path, mock_store: DualIndexStore) -> None:
    """Verify IngestionPipeline processes directory of .pdf, .md, and .txt files."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "doc1.md").write_text("Markdown passage for retrieval.", encoding="utf-8")
    (docs_dir / "doc2.txt").write_text("Text passage for retrieval.", encoding="utf-8")
    create_dummy_pdf(docs_dir / "doc3.pdf", ["PDF passage for retrieval."])
    (docs_dir / "ignored.csv").write_text("a,b\n1,2", encoding="utf-8")

    pipeline = IngestionPipeline(store=mock_store)
    result = pipeline.ingest_directory(docs_dir)

    assert result.files_processed == 3
    assert result.files_skipped == 0
    assert result.chunks_indexed == 3
    assert mock_store.count() == 3

    # Re-running on unchanged directory skips all 3
    re_result = pipeline.ingest_directory(docs_dir)
    assert re_result.files_processed == 0
    assert re_result.files_skipped == 3
    assert re_result.chunks_indexed == 0


# ============================================================================
# API Endpoint & CLI Integration Tests
# ============================================================================


def test_api_knowledge_ingest_multipart_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify POST /knowledge/ingest endpoint with multipart file upload."""
    client = TestClient(create_app())

    # Use in-memory or temp store for gateway testing
    temp_kb = tmp_path / "api_kb"
    temp_kb.mkdir()
    q_path = str(temp_kb / "qdrant")
    b_path = str(temp_kb / "bm25.joblib")
    isolated_store = DualIndexStore(
        qdrant_path=q_path,
        bm25_path=b_path,
        collection_name="api_test",
        embedder=MockEmbedder(),
    )

    from src.gateway import router

    monkeypatch.setattr(router, "get_knowledge_store", lambda: isolated_store)

    file_content = b"# Document for API ingestion\n\nTesting FastAPI file upload."
    response = client.post(
        "/knowledge/ingest",
        files={"file": ("upload.md", io.BytesIO(file_content), "text/markdown")},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_indexed"] >= 1
    assert "upload.md" in data["source_documents"]


def test_api_knowledge_ingest_json_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /knowledge/ingest endpoint with raw JSON text payload."""
    client = TestClient(create_app())

    temp_kb = tmp_path / "api_kb_json"
    temp_kb.mkdir()
    isolated_store = DualIndexStore(
        qdrant_path=str(temp_kb / "qdrant"),
        bm25_path=str(temp_kb / "bm25.joblib"),
        collection_name="api_test_json",
        embedder=MockEmbedder(),
    )

    from src.gateway import router

    monkeypatch.setattr(router, "get_knowledge_store", lambda: isolated_store)

    payload = {
        "text": "Payload knowledge text for direct API indexing.",
        "source_doc": "api_direct.txt",
    }
    response = client.post("/knowledge/ingest", json=payload)

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_indexed"] == 1
    assert "api_direct.txt" in data["source_documents"]


def test_cli_ingest_script(tmp_path: Path) -> None:
    """Verify python -m src.ingest --source <dir> runs as a valid CLI utility."""
    doc_file = tmp_path / "cli_sample.txt"
    doc_file.write_text("CLI ingestion test text for automated run.", encoding="utf-8")

    q_dir = tmp_path / "cli_qdrant"
    b_file = tmp_path / "cli_bm25.joblib"

    cmd = [
        sys.executable,
        "-m",
        "src.ingest",
        "--source",
        str(doc_file),
        "--qdrant-path",
        str(q_dir),
        "--bm25-path",
        str(b_file),
        "--mock-embedder",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI failed: {res.stderr}\n{res.stdout}"
    assert "Ingestion completed successfully" in res.stdout
    assert "Chunks indexed: 1" in res.stdout


def test_cli_ingest_main_direct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify src.ingest main() execution with directory target."""
    doc_dir = tmp_path / "cli_dir"
    doc_dir.mkdir()
    (doc_dir / "file1.txt").write_text("File 1 content for main.", encoding="utf-8")

    q_dir = tmp_path / "main_qdrant"
    b_file = tmp_path / "main_bm25.joblib"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest",
            "--source",
            str(doc_dir),
            "--qdrant-path",
            str(q_dir),
            "--bm25-path",
            str(b_file),
            "--mock-embedder",
        ],
    )
    from src.ingest import main

    main()


def test_cli_ingest_main_nonexistent_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify src.ingest main() exits with 1 on nonexistent source."""
    missing = tmp_path / "does_not_exist"
    monkeypatch.setattr(
        sys,
        "argv",
        ["ingest", "--source", str(missing)],
    )
    from src.ingest import main

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_api_knowledge_ingest_source_path_file_and_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify POST /knowledge/ingest with source_path pointing to a file and directory."""
    client = TestClient(create_app())
    isolated_store = DualIndexStore(
        in_memory=True,
        collection_name="api_test_source_path",
        embedder=MockEmbedder(),
    )
    from src.gateway import router

    monkeypatch.setattr(router, "get_knowledge_store", lambda: isolated_store)

    # Test file path
    f = tmp_path / "api_file.md"
    f.write_text("API ingested file content.", encoding="utf-8")
    resp_file = client.post("/knowledge/ingest", json={"source_path": str(f)})
    assert resp_file.status_code == 201
    assert resp_file.json()["chunks_indexed"] == 1

    # Test dir path
    d = tmp_path / "api_dir"
    d.mkdir()
    (d / "sub.txt").write_text("Sub document.", encoding="utf-8")
    resp_dir = client.post("/knowledge/ingest", json={"source_path": str(d)})
    assert resp_dir.status_code == 201

    # Test nonexistent path
    resp_missing = client.post("/knowledge/ingest", json={"source_path": str(tmp_path / "missing")})
    assert resp_missing.status_code == 404

    # Test empty JSON (neither text nor source_path)
    resp_empty = client.post("/knowledge/ingest", json={})
    assert resp_empty.status_code == 400


def test_api_knowledge_ingest_invalid_file_and_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify POST /knowledge/ingest error handling for bad uploads."""
    client = TestClient(create_app())
    isolated_store = DualIndexStore(
        in_memory=True,
        collection_name="api_test_err",
        embedder=MockEmbedder(),
    )
    from src.gateway import router

    monkeypatch.setattr(router, "get_knowledge_store", lambda: isolated_store)

    # Unsupported extension
    resp_bad_ext = client.post(
        "/knowledge/ingest",
        files={"file": ("data.csv", io.BytesIO(b"a,b"), "text/csv")},
    )
    assert resp_bad_ext.status_code == 400
    assert "Unsupported document format" in resp_bad_ext.json()["detail"]


def test_embeddinggemma_embedder_abstractions() -> None:
    """Verify EmbeddingGemmaEmbedder metadata, factory, and empty input handling."""
    embedder = EmbeddingGemmaEmbedder(dimension=768)
    assert embedder.dimension == 768
    assert embedder.model_name == "google/embeddinggemma-2"
    assert embedder.embed([]) == []
    assert embedder.embed_documents([]) == []
    assert embedder.embed_queries([]) == []

    default_gemma = get_default_embedder("google/embeddinggemma-2")
    assert isinstance(default_gemma, EmbeddingGemmaEmbedder)
    assert default_gemma.dimension == 768

    minilm = get_default_embedder("sentence-transformers/all-MiniLM-L6-v2", dimension=384)
    assert isinstance(minilm, FastEmbedEmbedder)
    assert minilm.dimension == 384
    assert minilm.embed([]) == []
    assert minilm.embed_documents([]) == []
    assert minilm.embed_queries([]) == []


def test_embedder_asymmetric_query_document_encoding() -> None:
    """Verify that MockEmbedder produces asymmetric vectors for queries vs documents."""
    mock = MockEmbedder(dimension=768)
    doc_vec = mock.embed_documents(["quantum computing"])[0]
    query_vec = mock.embed_query("quantum computing")
    assert len(doc_vec) == 768
    assert len(query_vec) == 768
    # Asymmetric encoding must produce different embeddings for retrieval separation
    assert doc_vec != query_vec


def test_store_dimension_migration_384_to_768(tmp_path: Path) -> None:
    """Verify ADR-0002 migration: 384-dim Qdrant store migrates automatically to 768-dim."""
    q_dir = tmp_path / "qdrant_migrate"
    b_file = tmp_path / "bm25_migrate.joblib"

    # Step 1: Initialize store with 384-dim mock embedder and insert chunks
    store_384 = DualIndexStore(
        qdrant_path=str(q_dir),
        bm25_path=str(b_file),
        collection_name="test_migration",
        embedder=MockEmbedder(dimension=384, model_name="legacy-minilm"),
    )
    chunk = EvidenceChunk(
        chunk_id="chunk-mig-01",
        text="Solar energy capacity expanded rapidly in 2024.",
        source_doc="doc_solar.txt",
        tier=2,
    )
    store_384.add_chunks([chunk])
    assert store_384.count() == 1
    meta_384 = store_384.get_collection_metadata()
    assert meta_384.get("dimension") == 384
    store_384.close()

    # Step 2: Open store on the same directory with 768-dim EmbeddingGemma embedder
    store_768 = DualIndexStore(
        qdrant_path=str(q_dir),
        bm25_path=str(b_file),
        collection_name="test_migration",
        embedder=MockEmbedder(dimension=768, model_name="google/embeddinggemma-2"),
    )
    # The collection should automatically detect dimension mismatch, recreate at 768, and re-embed
    assert store_768.count() == 1
    meta_768 = store_768.get_collection_metadata()
    assert meta_768.get("dimension") == 768
    assert meta_768.get("model_name") == "google/embeddinggemma-2"

    # Verify dense search functions on the migrated 768-dim collection
    results = store_768.search_dense("solar energy expansion", limit=5)
    assert len(results) == 1
    retrieved_chunk, score = results[0]
    assert retrieved_chunk.chunk_id == "chunk-mig-01"
    assert isinstance(score, float)
    store_768.close()


def test_deberta_nli_label_mapping() -> None:
    """Verify DeBERTa-v3-base label mapping follows documented order."""
    from src.schemas.verification import (
        DEBERTA_NLI_LABEL_ORDER,
        DEBERTA_TO_CLAIM_STATE,
        ClaimState,
        determine_claim_state_from_nli,
        map_nli_probabilities,
    )

    assert DEBERTA_NLI_LABEL_ORDER == ("contradiction", "entailment", "neutral")
    assert DEBERTA_TO_CLAIM_STATE["contradiction"] == ClaimState.CONTRADICTED
    assert DEBERTA_TO_CLAIM_STATE["entailment"] == ClaimState.SUPPORTED
    assert DEBERTA_TO_CLAIM_STATE["neutral"] == ClaimState.UNVERIFIED

    # Raw logits/probabilities from model: [0.05, 0.90, 0.05]
    # Index 0 is contradiction (0.05), Index 1 is entailment (0.90), Index 2 is neutral (0.05)
    probs = map_nli_probabilities([0.05, 0.90, 0.05])
    assert probs["contradiction"] == 0.05
    assert probs["entailment"] == 0.90
    assert probs["neutral"] == 0.05
    assert determine_claim_state_from_nli(probs) == ClaimState.SUPPORTED

    # Contradiction dominant: Index 0 is high
    contra_probs = map_nli_probabilities([0.88, 0.08, 0.04])
    assert determine_claim_state_from_nli(contra_probs) == ClaimState.CONTRADICTED

    # Neutral dominant: Index 2 is high
    neutral_probs = map_nli_probabilities([0.10, 0.10, 0.80])
    assert determine_claim_state_from_nli(neutral_probs) == ClaimState.UNVERIFIED

    # Length mismatch raises ValueError
    with pytest.raises(ValueError):
        map_nli_probabilities([0.5, 0.5])


def test_embeddinggemma_mock_dispatch_methods() -> None:
    """Verify EmbeddingGemmaEmbedder correctly invokes model methods or fallbacks."""

    class FakeModelWithMethods:
        def encode_document(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            return [[0.1] * 10 for _ in texts]

        def encode_query(self, queries: list[str], **kwargs: object) -> list[list[float]]:
            return [[0.9] * 10 for _ in queries]

    embedder = EmbeddingGemmaEmbedder(dimension=10)
    embedder._model = FakeModelWithMethods()

    docs = embedder.embed_documents(["doc text"])
    assert len(docs) == 1
    assert len(docs[0]) == 10
    assert docs[0][0] == 0.1

    queries = embedder.embed_queries(["query text"])
    assert len(queries) == 1
    assert len(queries[0]) == 10
    assert queries[0][0] == 0.9

    single = embedder.embed_query("single query text")
    assert len(single) == 10
    assert single[0] == 0.9
