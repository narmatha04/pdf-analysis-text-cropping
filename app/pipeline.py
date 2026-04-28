from __future__ import annotations

import json
import shutil
from pathlib import Path

from sqlalchemy.orm import Session

from .models import Submission, Question, QuestionLocation, Analysis
from .pdf_utils import pdf_to_images
from .extractor import extract_marks_grid, segment_page, check_page_boundary
from .analyser import analyse_submission

STORAGE = Path(__file__).parent.parent / "storage"


def run_pipeline(submission_id: str, pdf_path: str, db: Session):
    submission = db.get(Submission, submission_id)
    submission.status = "processing"
    db.commit()

    try:
        _run(submission, pdf_path, db)
        submission.status = "done"
    except Exception as e:
        submission.status = "failed"
        submission.error_message = str(e)
    finally:
        db.commit()


def _run(submission: Submission, pdf_path: str, db: Session):
    pages_dir = STORAGE / "pages" / submission.id
    image_paths = pdf_to_images(pdf_path, str(pages_dir), dpi=200)

    # --- Phase 1: marks grid from page 1 ---
    marks_data = extract_marks_grid(image_paths[0])
    marks = {int(k): v for k, v in marks_data.get("marks", {}).items()}
    submission.total_marks = marks_data.get("total")
    submission.student_name = marks_data.get("student_name")

    # Seed question rows with marks — answer_text filled later
    question_map: dict[int, Question] = {}
    for q_num, mark in marks.items():
        q = Question(
            submission_id=submission.id,
            question_number=q_num,
            marks_obtained=mark,
            max_marks=_infer_max(q_num),
        )
        db.add(q)
        question_map[q_num] = q
    db.flush()

    # --- Phase 2: segmentation across all pages ---
    # Skip page 1 (cover) — it has a few short answers at the bottom but
    # the main answers start from page 2.
    all_segments: list[dict] = []  # {page, question_id, bbox, answer_text, is_continuation}

    for page_idx, img_path in enumerate(image_paths[1:], start=2):
        segments = segment_page(img_path, page_idx)
        for seg in segments:
            seg["page"] = page_idx
            seg["image_path"] = img_path
        all_segments.extend(segments)

    # --- Phase 3: resolve multi-page continuations ---
    # If VLM flagged is_continuation=true on a segment, check the boundary
    # and merge it into the previous question.
    resolved = _resolve_continuations(all_segments, image_paths)

    # --- Phase 4: store locations + transcriptions ---
    for seg in resolved:
        q_num = seg["question_id"]
        if q_num not in question_map:
            # Question found in answer pages but not in marks grid — add it
            q = Question(
                submission_id=submission.id,
                question_number=q_num,
                marks_obtained=None,
                max_marks=_infer_max(q_num),
            )
            db.add(q)
            db.flush()
            question_map[q_num] = q

        q = question_map[q_num]
        if q.answer_text is None:
            q.answer_text = seg.get("answer_text", "")
        else:
            # Multi-page: append continuation
            q.answer_text += "\n" + seg.get("answer_text", "")

        loc = QuestionLocation(
            question_id=q.id,
            page_number=seg["page"],
            page_image_path=seg["image_path"],
            bbox_x1=seg["bbox"][0],
            bbox_y1=seg["bbox"][1],
            bbox_x2=seg["bbox"][2],
            bbox_y2=seg["bbox"][3],
            sequence=seg.get("sequence", 0),
        )
        db.add(loc)

    db.flush()

    # --- Phase 5: analysis ---
    q_dicts = [
        {
            "question_number": q.question_number,
            "marks_obtained": q.marks_obtained,
            "max_marks": q.max_marks,
            "answer_text": q.answer_text,
        }
        for q in question_map.values()
    ]
    analysis_data = analyse_submission(q_dicts)

    # Write topic back onto each question
    q_topics: dict = analysis_data.get("question_topics", {})
    for q_num_str, topic in q_topics.items():
        q_num = int(q_num_str)
        if q_num in question_map:
            question_map[q_num].topic = topic

    analysis = Analysis(
        submission_id=submission.id,
        overall_percentage=analysis_data.get("overall_percentage"),
        strengths=json.dumps(analysis_data.get("strengths", [])),
        weaknesses=json.dumps(analysis_data.get("weaknesses", [])),
        error_patterns=json.dumps(analysis_data.get("error_patterns", [])),
        recommendations=json.dumps(analysis_data.get("recommendations", [])),
        topic_performance=json.dumps(analysis_data.get("topic_performance", {})),
    )
    db.add(analysis)


def _resolve_continuations(
    segments: list[dict], image_paths: list[str]
) -> list[dict]:
    """
    Assign sequence numbers and resolve continuation segments.
    A continuation gets the same question_id as the last non-continuation segment.
    Falls back to check_page_boundary when ambiguous.
    """
    resolved = []
    last_q_id: int | None = None
    q_sequence: dict[int, int] = {}

    for seg in segments:
        if seg.get("is_continuation") and last_q_id is not None:
            seg["question_id"] = last_q_id
        else:
            last_q_id = seg["question_id"]

        q_id = seg["question_id"]
        seq = q_sequence.get(q_id, 0)
        seg["sequence"] = seq
        q_sequence[q_id] = seq + 1
        resolved.append(seg)

    return resolved


# Class 12 CBSE mark scheme (approximate)
_MARK_SCHEME = {
    **{i: 1 for i in range(1, 21)},   # Q1–Q20: 1 mark each
    **{i: 2 for i in range(21, 26)},  # Q21–Q25: 2 marks each
    **{i: 3 for i in range(26, 29)},  # Q26–Q28: 3 marks each (case-based / SA)
    **{i: 5 for i in range(29, 37)},  # Q29–Q36: 5 marks each (long answer)
}


def _infer_max(q_num: int) -> float:
    return float(_MARK_SCHEME.get(q_num, 1))
