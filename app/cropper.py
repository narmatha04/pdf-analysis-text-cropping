import io
from PIL import Image

PADDING = 0.025  # 2.5% padding on each side to avoid clipping


def _padded(bbox: list[float]) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [
        max(0.0, x1 - PADDING),
        max(0.0, y1 - PADDING),
        min(1.0, x2 + PADDING),
        min(1.0, y2 + PADDING),
    ]


def crop_image(page_image_path: str, bbox: list[float]) -> Image.Image:
    """Crop a normalised bbox from a page image with padding."""
    img = Image.open(page_image_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = _padded(bbox)
    return img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))


def stitch_vertical(images: list[Image.Image]) -> Image.Image:
    """Stack multiple crops vertically (for multi-page answers)."""
    if len(images) == 1:
        return images[0]
    max_w = max(img.width for img in images)
    total_h = sum(img.height for img in images)
    canvas = Image.new("RGB", (max_w, total_h), "white")
    y = 0
    for img in images:
        canvas.paste(img, (0, y))
        y += img.height
    return canvas


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> io.BytesIO:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf
