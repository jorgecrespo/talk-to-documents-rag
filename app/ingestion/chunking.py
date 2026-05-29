import json
from dataclasses import dataclass
from pathlib import Path

from .loader import PageText


@dataclass(frozen=True)
class Chunk:
    source_file: str
    page_number: int
    chunk_index: int
    text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "text": self.text,
        }


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into simple overlapping chunks."""

    clean_text = " ".join(text.split())
    if not clean_text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must be zero or greater")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))
        chunks.append(clean_text[start:end])
        if end == len(clean_text):
            break
        start = end - overlap

    return chunks


def chunk_pages(pages: list[PageText], chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]:
    """Chunk extracted pages and keep their source metadata."""

    output: list[Chunk] = []
    for page in pages:
        for chunk_index, chunk in enumerate(chunk_text(page.text, chunk_size=chunk_size, overlap=overlap), start=1):
            output.append(
                Chunk(
                    source_file=page.source_file,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=chunk,
                )
            )
    return output


def save_chunks_to_json(chunks: list[Chunk], output_path: str | Path) -> Path:
    """Save chunks to a JSON file for inspection."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return path
