import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PageText:
    source_file: str
    page_number: int
    text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_file": self.source_file,
            "page_number": self.page_number,
            "text": self.text,
        }


def extract_pdf_pages(pdf_path: str | Path) -> list[PageText]:
    """Extract text from each page in a PDF.

    Empty pages are skipped.
    """

    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
        raise ModuleNotFoundError(
            "pypdf is required to extract PDFs. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    path = Path(pdf_path)
    reader = PdfReader(str(path))

    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        pages.append(PageText(source_file=path.name, page_number=index, text=text))

    return pages


def load_pdfs_from_dir(raw_dir: str | Path) -> list[PageText]:
    """Load all PDFs from a directory and return extracted page text."""

    directory = Path(raw_dir)
    pages: list[PageText] = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        pages.extend(extract_pdf_pages(pdf_path))
    return pages


def save_pages_to_json(pages: list[PageText], output_path: str | Path) -> Path:
    """Save extracted pages to a JSON file for inspection."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([page.to_dict() for page in pages], ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return path
