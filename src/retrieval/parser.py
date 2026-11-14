"""Document text extraction and hashing utilities.

Supports PDF, Markdown (.md), and plain text (.txt) files (REQ-RET-01).
"""

import hashlib
import io
from pathlib import Path
from typing import Union
from pypdf import PdfReader


def compute_file_hash(file_path: Union[Path, str]) -> str:
    """Calculate the SHA-256 hash of a file on disk."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_content_hash(content: str) -> str:
    """Calculate the SHA-256 hash of raw string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def extract_text_from_bytes(data: bytes, filename: str) -> str:
    """Extract plain text from in-memory document bytes.

    Args:
        data: Raw document bytes.
        filename: File name used to detect document type.

    Returns:
        Extracted plain text.

    Raises:
        ValueError: If file extension is unsupported or content is malformed.
    """
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""

    if ext in {".md", ".markdown", ".txt"}:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="replace")

    elif ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            extracted_pages = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_pages.append(page_text)
            return "\n\n".join(extracted_pages)
        except Exception as e:
            raise ValueError(f"Failed to parse PDF document bytes: {e}") from e

    else:
        raise ValueError(f"Unsupported document format: {ext}. Supported formats: .pdf, .md, .txt")


def extract_text_from_file(file_path: Union[Path, str]) -> str:
    """Extract plain text from PDF, Markdown, and TXT files.

    Args:
        file_path: Path to the target document.

    Returns:
        Extracted raw text content.

    Raises:
        FileNotFoundError: If the document path does not exist.
        ValueError: If the file extension is not supported (.pdf, .md, .txt).
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    raw_bytes = path.read_bytes()
    return extract_text_from_bytes(raw_bytes, path.name)
