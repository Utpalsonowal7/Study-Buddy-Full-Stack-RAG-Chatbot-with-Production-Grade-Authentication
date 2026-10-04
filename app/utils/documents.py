from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePath
import re

from fastapi import HTTPException
from pypdf import PdfReader


@dataclass
class TextChunk:
    text: str
    page: int | None


def safe_filename(filename: str | None) -> str:
    name = PurePath((filename or "document").replace("\\", "/")).name
    name = re.sub(r"[^\w. -]", "_", name)[:255]
    if not name or name in {".", ".."}:
        raise HTTPException(400, "A valid filename is required.")
    return name


def extract_chunks(data: bytes, filename: str, limit: int = 200) -> tuple[str, list[TextChunk]]:
    extension = PurePath(filename).suffix.lower()
    if extension == ".pdf":
        content_type = "application/pdf"
        if not data.startswith(b"%PDF-"):
            raise HTTPException(400, "The file is not a valid PDF.")
        try:
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                raise HTTPException(400, "Encrypted PDFs are not supported.")
            if len(reader.pages) > 200:
                raise HTTPException(413, "PDFs may contain at most 200 pages.")
            pages = [(page.extract_text() or "", index + 1) for index, page in enumerate(reader.pages)]
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(400, "The PDF could not be read.") from exc
    elif extension in {".txt", ".md"}:
        content_type = "text/plain" if extension == ".txt" else "text/markdown"
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(400, "Text files must use UTF-8 encoding.") from exc
        if "\x00" in text:
            raise HTTPException(400, "Binary content is not supported.")
        pages = [(text, None)]
    else:
        raise HTTPException(415, "Supported files: PDF, TXT, and Markdown.")

    chunks = []
    for text, page in pages:
        text = re.sub(r"\s+", " ", text).strip()
        start = 0
        while start < len(text):
            end = min(start + 1800, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start + 900, end)
                if boundary > start:
                    end = boundary
            chunks.append(TextChunk(text[start:end], page))
            if len(chunks) > limit:
                raise HTTPException(413, f"Document exceeds the {limit}-chunk indexing limit.")
            if end == len(text):
                break
            start = end - 200
    if not chunks:
        raise HTTPException(422, "No readable text found. Scanned PDFs require OCR before upload.")
    return content_type, chunks
