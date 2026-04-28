from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import anthropic

client = anthropic.Anthropic()


def _encode(image_path: str) -> str:
    return base64.standard_b64encode(Path(image_path).read_bytes()).decode("utf-8")


def _parse_json(text: str) -> dict | list:
    text = text.strip()
    # strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # fallback: grab outermost { } or [ ]
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise


def extract_marks_grid(page1_image_path: str) -> dict:
    """
    Read the printed marks table on the cover page.
    Returns {"marks": {"1": 0, "2": 1, ...}, "total": 34, "student_name": "..."}
    Marks may be 0, 0.5, 1, 2, 3, 4, 5 or null (not attempted).
    """
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _encode(page1_image_path),
                    },
                },
                {
                    "type": "text",
                    "text": (
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
                    ),
                },
            ],
        }],
    )
    return _parse_json(response.content[0].text)


def segment_page(page_image_path: str, page_number: int) -> list[dict]:
    """
    Identify every question on a page and return bounding boxes.

    Based on the actual paper structure:
    - Ruled lined paper, single column
    - Question numbers handwritten in left margin (may be circled or have parentheses)
    - Blue/black student ink; red evaluator marks (ignore red)
    - Some pages have 2D matrix notation spreading horizontally
    - Last pages may be supplementary sheets

    Returns list of:
    {
      "question_id": 21,
      "bbox": [x1, y1, x2, y2],   # normalised 0-1
      "answer_text": "...",
      "has_worked_steps": true,
      "confidence": "high" | "medium" | "low",
      "is_continuation": false      # true if this is continuation of prev page's answer
    }
    """
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _encode(page_image_path),
                    },
                },
                {
                    "type": "text",
                    "text": (
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
                    ),
                },
            ],
        }],
    )
    raw = _parse_json(response.content[0].text)
    return raw.get("questions", [])


def check_page_boundary(page_a_path: str, page_b_path: str, page_a_number: int) -> dict | None:
    """
    Send the bottom of page_a and top of page_b to detect if an answer continues.
    Returns {"question_id": 28, "continues_on_next": true} or None.
    Only called when segmentation flags a continuation.
    """
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _encode(page_a_path),
                    },
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _encode(page_b_path),
                    },
                },
                {
                    "type": "text",
                    "text": (
                        f"Image 1 is page {page_a_number}, Image 2 is page {page_a_number + 1} "
                        "of a student answer sheet.\n"
                        "Does the last answer on page 1 continue onto page 2 "
                        "(i.e. page 2 starts mid-answer with no new question number)?\n"
                        "If yes, what is the question_id of the continuing answer?\n\n"
                        "Respond ONLY with JSON: "
                        '{"continues": true, "question_id": 28} or {"continues": false}'
                    ),
                },
            ],
        }],
    )
    result = _parse_json(response.content[0].text)
    return result if result.get("continues") else None
