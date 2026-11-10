import pytest

from app.parsing.pdf import NoExtractableTextError, PDFParseError, extract_pdf_text
from tests.pdf_factory import image_only_pdf, text_pdf


def test_extracts_text_from_pdf() -> None:
    data = text_pdf(["Backend Engineer", "Skills: Python, FastAPI"])
    text = extract_pdf_text(data)
    assert "Backend Engineer" in text
    assert "Python, FastAPI" in text


def test_image_only_pdf_raises_no_text() -> None:
    with pytest.raises(NoExtractableTextError, match="no extractable text"):
        extract_pdf_text(image_only_pdf())


def test_invalid_bytes_raise_parse_error() -> None:
    with pytest.raises(PDFParseError):
        extract_pdf_text(b"this is not a pdf")
