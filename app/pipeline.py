from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from .models import Submission, Question, QuestionLocation, Analysis, SessionLocal
from .pdf_utils import pdf_to_images
from .extractor import extract_marks_grid, segment_page
from .analyser import analyse_submission

STORAGE = Path(__file__).parent.parent / "storage"


def run_pipeline(submission_id: str, pdf_path: str):
    """
    Entry point for background task.
    Creates its own DB session — never reuse the request-scoped session.
    Runs three independently committed phases so a failure at phase N
    doesn't require re-running phases 1..N-1.
    """
    db = SessionLocal()
    try:
        submission = db.get(Submission, submission_id)
        submission.status = "processing"
        db.commit()

        pages_dir = STORAGE / "pages" / submission_id
        image_paths = pdf_to_images(pdf_path, str(pages_dir), dpi=150)

        _phase1_marks(submission, image_paths[0], db)
        _phase2_segmentation(submission, image_paths[1:], db)
        _phase3_analysis(submission, db)

        submission.status = "done"
        db.commit()
        print(f"[pipeline] submission {submission_id} complete")

    except Exception as e:
        db.rollback()
        try:
            submission = db.get(Submission, submission_id)
            if submission:
                submission.status = "failed"
                submission.error_message = str(e)
                db.commit()
        except Exception:
            pass
        print(f"[pipeline] FAILED: {e}")
        raise
    finally:
        db.close()


# ── Phase 1: marks grid ───────────────────────────────────────────────────────

def _phase1_marks(submission: Submission, page1_path: str, db: Session):
    """
    Extract marks grid from cover page and seed question rows.
    SKIP if questions already exist for this submission (resume).
    """
    existing = db.query(Question).filter_by(submission_id=submission.id).count()
    if existing > 0:
        print(f"[pipeline] phase 1 already done ({existing} questions) — skipping")
        return

    print("[pipeline] phase 1: extracting marks grid")
    marks_data = extract_marks_grid(page1_path)

    marks = {}
    for k, v in marks_data.get("marks", {}).items():
        try:
            marks[int(k)] = v
        except (TypeError, ValueError):
            pass

    submission.total_marks = marks_data.get("total")
    submission.student_name = marks_data.get("student_name")

    for q_num, mark in marks.items():
        db.add(Question(
            submission_id=submission.id,
            question_number=q_num,
            marks_obtained=mark,
            max_marks=_infer_max(q_num),
        ))

    db.commit()
    print(f"[pipeline] phase 1 done — {len(marks)} questions seeded")


# ── Phase 2: segmentation ─────────────────────────────────────────────────────

def _phase2_segmentation(submission: Submission, answer_image_paths: list[str], db: Session):
    """
    Segment each answer page and store bounding boxes + transcriptions.
    SKIP pages that already have locations stored (resume from last saved page).
    """
    # Build question map from DB
    questions = db.query(Question).filter_by(submission_id=submission.id).all()
    question_map: dict[int, Question] = {q.question_number: q for q in questions}

    # Find which pages already have locations saved
    done_pages: set[int] = set()
    for q in questions:
        for loc in q.locations:
            done_pages.add(loc.page_number)

    all_segments: list[dict] = []

    for i, img_path in enumerate(answer_image_paths):
        page_idx = i + 2  # pages are 1-indexed, page 1 is cover
        if page_idx in done_pages:
            print(f"[pipeline] page {page_idx} already segmented — skipping")
            continue
        try:
            segments = segment_page(img_path, page_idx)
        except Exception as e:
            print(f"[pipeline] page {page_idx} failed: {e} — skipping")
            segments = []
        for seg in segments:
            seg["page"] = page_idx
            seg["image_path"] = img_path
        all_segments.extend(segments)
        print(f"[pipeline] page {page_idx} done — {len(segments)} questions found")

    # Resolve continuations
    resolved = _resolve_continuations(all_segments)

    # Store locations and transcriptions
    for seg in resolved:
        q_num = seg.get("question_id")
        if q_num is None:
            continue
        try:
            q_num = int(q_num)
        except (TypeError, ValueError):
            continue

        if q_num not in question_map:
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
        q.answer_text = (q.answer_text or "") + (
            "\n" + seg.get("answer_text", "") if q.answer_text else seg.get("answer_text", "")
        )

        bbox = seg.get("bbox", [0, 0, 1, 1])
        db.add(QuestionLocation(
            question_id=q.id,
            page_number=seg["page"],
            page_image_path=seg["image_path"],
            bbox_x1=bbox[0],
            bbox_y1=bbox[1],
            bbox_x2=bbox[2],
            bbox_y2=bbox[3],
            sequence=seg.get("sequence", 0),
        ))

    db.commit()
    print("[pipeline] phase 2 done — all segments stored")


# ── Phase 3: analysis ─────────────────────────────────────────────────────────

def _phase3_analysis(submission: Submission, db: Session):
    """
    Run strengths/weaknesses analysis over all transcribed questions.
    SKIP if analysis already exists for this submission (resume).
    """
    if db.query(Analysis).filter_by(submission_id=submission.id).count() > 0:
        print("[pipeline] phase 3 already done — skipping")
        return

    print("[pipeline] phase 3: running analysis")
    questions = db.query(Question).filter_by(submission_id=submission.id).all()
    q_dicts = [
        {
            "question_number": q.question_number,
            "marks_obtained": q.marks_obtained,
            "max_marks": q.max_marks,
            "answer_text": q.answer_text,
        }
        for q in questions
    ]

    analysis_data = analyse_submission(q_dicts)

    # Write topic back onto each question
    for q_num_str, topic in analysis_data.get("question_topics", {}).items():
        try:
            q_num = int(q_num_str)
            q = next((q for q in questions if q.question_number == q_num), None)
            if q:
                q.topic = topic
        except (TypeError, ValueError):
            pass

    db.add(Analysis(
        submission_id=submission.id,
        overall_percentage=analysis_data.get("overall_percentage"),
        strengths=json.dumps(analysis_data.get("strengths", [])),
        weaknesses=json.dumps(analysis_data.get("weaknesses", [])),
        error_patterns=json.dumps(analysis_data.get("error_patterns", [])),
        recommendations=json.dumps(analysis_data.get("recommendations", [])),
        topic_performance=json.dumps(analysis_data.get("topic_performance", {})),
    ))
    db.commit()
    print("[pipeline] phase 3 done — analysis stored")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_continuations(segments: list[dict]) -> list[dict]:
    resolved = []
    last_q_id: int | None = None
    q_sequence: dict[int, int] = {}

    for seg in segments:
        if seg.get("is_continuation") and last_q_id is not None:
            seg["question_id"] = last_q_id
        else:
            last_q_id = seg.get("question_id")

        q_id = seg.get("question_id")
        if q_id is not None:
            seq = q_sequence.get(q_id, 0)
            seg["sequence"] = seq
            q_sequence[q_id] = seq + 1
        resolved.append(seg)

    return resolved


_MARK_SCHEME = {
    **{i: 1 for i in range(1, 21)},
    **{i: 2 for i in range(21, 26)},
    **{i: 3 for i in range(26, 29)},
    **{i: 5 for i in range(29, 37)},
}


def _infer_max(q_num: int) -> float:
    return float(_MARK_SCHEME.get(q_num, 1))
