import json
import shutil
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .models import init_db, get_db, Submission, Question, QuestionLocation
from .pipeline import run_pipeline
from .cropper import crop_image, stitch_vertical, image_to_bytes

STORAGE = Path(__file__).parent.parent / "storage"
STORAGE.mkdir(exist_ok=True)
(STORAGE / "uploads").mkdir(exist_ok=True)

app = FastAPI(title="Answer Sheet Analyser")


@app.on_event("startup")
def startup():
    init_db()


# ── 1. Upload PDF ─────────────────────────────────────────────────────────────

@app.post("/submissions/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    sub_id = str(uuid.uuid4())
    dest = STORAGE / "uploads" / f"{sub_id}.pdf"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    submission = Submission(id=sub_id, pdf_path=str(dest))
    db.add(submission)
    db.commit()

    # Run pipeline in background — upload returns immediately
    # NOTE: do NOT pass db here — background task must open its own session
    background_tasks.add_task(run_pipeline, sub_id, str(dest))

    return {"submission_id": sub_id, "status": "processing"}


# ── 2. Status ─────────────────────────────────────────────────────────────────

@app.get("/submissions/{sub_id}/status")
def get_status(sub_id: str, db: Session = Depends(get_db)):
    sub = _get_or_404(db, Submission, sub_id)
    return {
        "submission_id": sub_id,
        "status": sub.status,
        "error": sub.error_message,
    }


# ── 3. Marks grid ─────────────────────────────────────────────────────────────

@app.get("/submissions/{sub_id}/marks")
def get_marks(sub_id: str, db: Session = Depends(get_db)):
    _get_or_404(db, Submission, sub_id)
    questions = db.query(Question).filter_by(submission_id=sub_id).all()
    return {
        "submission_id": sub_id,
        "marks": {q.question_number: q.marks_obtained for q in questions},
        "total": sum(q.marks_obtained or 0 for q in questions),
    }


# ── 4. Analysis ───────────────────────────────────────────────────────────────

@app.get("/submissions/{sub_id}/analysis")
def get_analysis(sub_id: str, db: Session = Depends(get_db)):
    sub = _get_or_404(db, Submission, sub_id)
    if not sub.analysis:
        raise HTTPException(status_code=404, detail="Analysis not ready yet")
    a = sub.analysis
    return {
        "submission_id": sub_id,
        "overall_percentage": a.overall_percentage,
        "topic_performance": json.loads(a.topic_performance or "{}"),
        "strengths": json.loads(a.strengths or "[]"),
        "weaknesses": json.loads(a.weaknesses or "[]"),
        "error_patterns": json.loads(a.error_patterns or "[]"),
        "recommendations": json.loads(a.recommendations or "[]"),
    }


# ── 5. All questions ──────────────────────────────────────────────────────────

@app.get("/submissions/{sub_id}/questions")
def list_questions(sub_id: str, db: Session = Depends(get_db)):
    _get_or_404(db, Submission, sub_id)
    questions = (
        db.query(Question)
        .filter_by(submission_id=sub_id)
        .order_by(Question.question_number)
        .all()
    )
    return [
        {
            "question_number": q.question_number,
            "marks_obtained": q.marks_obtained,
            "max_marks": q.max_marks,
            "topic": q.topic,
            "has_answer": q.answer_text is not None,
        }
        for q in questions
    ]


# ── 6. Single question detail ─────────────────────────────────────────────────

@app.get("/submissions/{sub_id}/questions/{q_num}")
def get_question(sub_id: str, q_num: int, db: Session = Depends(get_db)):
    q = _get_question_or_404(db, sub_id, q_num)
    return {
        "question_number": q.question_number,
        "marks_obtained": q.marks_obtained,
        "max_marks": q.max_marks,
        "topic": q.topic,
        "answer_text": q.answer_text,
        "error_analysis": json.loads(q.error_analysis or "{}"),
        "pages": [
            {
                "page_number": loc.page_number,
                "bbox": [loc.bbox_x1, loc.bbox_y1, loc.bbox_x2, loc.bbox_y2],
                "sequence": loc.sequence,
            }
            for loc in q.locations
        ],
    }


# ── 7. Crop image ─────────────────────────────────────────────────────────────

@app.get("/submissions/{sub_id}/questions/{q_num}/crop")
def get_crop(sub_id: str, q_num: int, db: Session = Depends(get_db)):
    q = _get_question_or_404(db, sub_id, q_num)
    if not q.locations:
        raise HTTPException(status_code=404, detail="No location data for this question")

    crops = [
        crop_image(loc.page_image_path, [loc.bbox_x1, loc.bbox_y1, loc.bbox_x2, loc.bbox_y2])
        for loc in q.locations
    ]
    final = stitch_vertical(crops)
    return StreamingResponse(image_to_bytes(final), media_type="image/png")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_or_404(db, model, id_):
    obj = db.get(model, id_)
    if not obj:
        raise HTTPException(status_code=404, detail=f"{model.__name__} not found")
    return obj


def _get_question_or_404(db: Session, sub_id: str, q_num: int) -> Question:
    q = (
        db.query(Question)
        .filter_by(submission_id=sub_id, question_number=q_num)
        .first()
    )
    if not q:
        raise HTTPException(status_code=404, detail=f"Question {q_num} not found")
    return q
