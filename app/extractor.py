from __future__ import annotations

import json
import os
import re
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=60000),  # 60s timeout per call
)

_MODEL = "models/gemini-2.5-flash"


def _load_image(path: str) -> Image.Image:
    return Image.open(path)


def _parse_json(text: str) -> dict | list:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise


def extract_marks_grid(page1_image_path: str) -> dict:
    """
    Read the printed marks table on the cover page.
    Returns {"marks": {"1": 0, "2": 1, ...}, "total": 34, "student_name": "..."}
    """
    img = _load_image(page1_image_path)

    prompt = (
        "This is the cover page of a Class 12 student answer sheet.\n"
        "There is a printed marks table with Question No. and Marks Obtained rows.\n"
        "Extract every question number and its marks obtained.\n"
        "Rules:\n"
        "- Fractions like 1/2 or 1½ → convert to decimal (0.5, 1.5)\n"
        "- A dash or blank → null (not attempted)\n"
        "- Also extract the student name if visible\n"
        "- Also extract the circled TOTAL marks at top right\n\n"
        "Respond ONLY with valid JSON, no explanation:\n"
        '{"marks": {"1": 0, "2": 1, ...}, "total": 34, "student_name": null}'
    )

    response = client.models.generate_content(
        model=_MODEL,
        contents=[img, prompt],
    )
    return _parse_json(response.text)


def segment_page(page_image_path: str, page_number: int) -> list[dict]:
    """
    Identify every question on a page and return bounding boxes.

    Returns list of:
    {
      "question_id": 21,
      "bbox": [x1, y1, x2, y2],   # normalised 0-1
      "answer_text": "...",
      "has_worked_steps": true,
      "confidence": "high" | "medium" | "low",
      "is_continuation": false
    }
    """
    img = _load_image(page_image_path)

    prompt = (
        f"This is page {page_number} of a Class 12 Maths handwritten answer sheet.\n"
        "The paper is ruled/lined. Student writes answers one after another, top to bottom.\n"
        "Question numbers appear in the LEFT MARGIN, handwritten — may be circled, have brackets, or written as '21)' or 'Q21'.\n"
        "Evaluator marks in RED ink (ticks, crosses, partial marks) — IGNORE these.\n"
        "Transcribe only BLUE or BLACK student handwriting.\n\n"
        "For EACH question visible on this page:\n"
        "1. question_id: the question number as an integer\n"
        "2. bbox: [x1, y1, x2, y2] normalised 0-1, covering ALL content for that question\n"
        "   - Include full width if matrices or wide expressions are present\n"
        "   - Add a small buffer so no content is cut off\n"
        "3. answer_text: transcribe what the student wrote\n"
        "   - Use plain text maths: x^2, sqrt(x), integral, (x-1)/(x+2)\n"
        "   - For matrices write [[a,b],[c,d]]\n"
        "4. has_worked_steps: true if multi-line working is shown\n"
        "5. confidence: 'high' if question number is clearly visible, 'medium' if inferred, 'low' if guessed\n"
        "6. is_continuation: true ONLY if this looks like the continuation of an answer\n"
        "   that started on the previous page (no question number visible for it)\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"page": ' + str(page_number) + ', "questions": ['
        '{"question_id": 1, "bbox": [0.0, 0.1, 1.0, 0.3], '
        '"answer_text": "...", "has_worked_steps": false, '
        '"confidence": "high", "is_continuation": false}'
        "]}"
    )

    response = client.models.generate_content(
        model=_MODEL,
        contents=[img, prompt],
    )
    raw = _parse_json(response.text)
    return raw.get("questions", [])


def check_page_boundary(page_a_path: str, page_b_path: str, page_a_number: int) -> dict | None:
    """
    Detect if an answer continues across two pages.
    Returns {"question_id": 28, "continues": true} or None.
    """
    img_a = _load_image(page_a_path)
    img_b = _load_image(page_b_path)

    prompt = (
        f"Image 1 is page {page_a_number}, Image 2 is page {page_a_number + 1} "
        "of a student answer sheet.\n"
        "Does the last answer on page 1 continue onto page 2 "
        "(i.e. page 2 starts mid-answer with no new question number)?\n"
        "If yes, what is the question_id of the continuing answer?\n\n"
        "Respond ONLY with JSON: "
        '{"continues": true, "question_id": 28} or {"continues": false}'
    )

    response = client.models.generate_content(
        model=_MODEL,
        contents=[img_a, img_b, prompt],
    )
    result = _parse_json(response.text)
    return result if result.get("continues") else None
