"""
PaddleOCR wrapper — extracts raw text from invoice images.
Text is used as a hint/fallback for the VLM model.
"""

from __future__ import annotations

_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR

        _ocr = PaddleOCR(use_angle_cls=True, lang="vi", show_log=False)
    return _ocr


def extract_text(image_path: str) -> str:
    """
    Run PaddleOCR on an image and return concatenated text.
    Preserves line structure for better VLM understanding.
    """
    ocr = _get_ocr()
    result = ocr.ocr(image_path, cls=True)

    if not result or not result[0]:
        return ""

    lines = []
    for line in result[0]:
        text = line[1][0]
        confidence = line[1][1]
        if confidence > 0.5:  # filter low confidence
            lines.append(text)

    return "\n".join(lines)


def extract_text_with_bbox(image_path: str) -> list[dict]:
    """
    Return text with bounding box info for spatial reasoning.
    """
    ocr = _get_ocr()
    result = ocr.ocr(image_path, cls=True)

    if not result or not result[0]:
        return []

    items = []
    for line in result[0]:
        bbox, (text, conf) = line
        if conf > 0.5:
            items.append({"text": text, "confidence": conf, "bbox": bbox})

    return items
