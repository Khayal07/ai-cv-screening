from fpdf import FPDF
from PIL import Image


def text_pdf(lines: list[str]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for line in lines:
        pdf.cell(text=line or " ", new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def image_only_pdf() -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.image(Image.new("RGB", (200, 200), "gray"), x=10, y=10, w=80)
    return bytes(pdf.output())
