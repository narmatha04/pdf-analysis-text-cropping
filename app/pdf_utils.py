import fitz
from pathlib import Path


def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 200) -> list[str]:
    """
    Rasterise every page of a PDF to PNG.
    Handles page-level rotation (page 1 of sample_paper_H is 90°).
    Returns ordered list of absolute image paths.
    """
    doc = fitz.open(pdf_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72, dpi / 72).prerotate(-page.rotation)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        path = out / f"page_{i + 1:03d}.png"
        pix.save(str(path))
        paths.append(str(path))

    return paths


def page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    return doc.page_count
