"""Text-only PDF extraction. Embedded images (such as photos) are never read."""

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFParseError(Exception):
    """Raised when the file cannot be read as a PDF."""


class NoExtractableTextError(PDFParseError):
    """Raised when a PDF contains no text layer, e.g. a scanned image."""


def extract_pdf_text(data: bytes) -> str:
    pages = _read_pages(data)
    text = "\n".join(page.strip() for page in pages if page.strip())
    if not text:
        raise NoExtractableTextError("no extractable text")
    return text


def _read_pages(data: bytes) -> list[str]:
    try:
        reader = PdfReader(BytesIO(data))
        return [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError, KeyError) as exc:
        raise PDFParseError(f"invalid PDF: {exc}") from exc
