import uuid
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    DateTime, ForeignKey, Boolean, Text
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


def _uuid():
    return str(uuid.uuid4())


class Submission(Base):
    __tablename__ = "submissions"

    id = Column(String, primary_key=True, default=_uuid)
    pdf_path = Column(String, nullable=False)
    student_name = Column(String)
    total_marks = Column(Float)
    max_marks = Column(Float)
    status = Column(String, default="pending")  # pending | processing | done | failed
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    questions = relationship("Question", back_populates="submission", cascade="all, delete-orphan")
    analysis = relationship("Analysis", back_populates="submission", uselist=False, cascade="all, delete-orphan")


class Question(Base):
    __tablename__ = "questions"

    id = Column(String, primary_key=True, default=_uuid)
    submission_id = Column(String, ForeignKey("submissions.id"), nullable=False)
    question_number = Column(Integer, nullable=False)
    marks_obtained = Column(Float)   # from marks grid
    max_marks = Column(Float)
    answer_text = Column(Text)       # VLM transcription
    topic = Column(String)
    error_analysis = Column(Text)    # JSON string
    is_supplementary = Column(Boolean, default=False)

    submission = relationship("Submission", back_populates="questions")
    locations = relationship("QuestionLocation", back_populates="question",
                             cascade="all, delete-orphan", order_by="QuestionLocation.sequence")


class QuestionLocation(Base):
    """One row per page-segment a question occupies. Multi-page = multiple rows."""
    __tablename__ = "question_locations"

    id = Column(String, primary_key=True, default=_uuid)
    question_id = Column(String, ForeignKey("questions.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    page_image_path = Column(String, nullable=False)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    sequence = Column(Integer, default=0)  # order within multi-page answer

    question = relationship("Question", back_populates="locations")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String, primary_key=True, default=_uuid)
    submission_id = Column(String, ForeignKey("submissions.id"), nullable=False, unique=True)
    overall_percentage = Column(Float)
    strengths = Column(Text)     # JSON
    weaknesses = Column(Text)    # JSON
    error_patterns = Column(Text)  # JSON
    recommendations = Column(Text) # JSON
    topic_performance = Column(Text)  # JSON

    submission = relationship("Submission", back_populates="analysis")


# SQLite for dev
engine = create_engine(
    "sqlite:///./answer_analyser.db",
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
