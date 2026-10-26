"""SQLAlchemy models: every prediction and every piece of administrator feedback is stored."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    unit = Column(String(32), index=True); lead_time = Column(Integer); lead_time_used = Column(Integer)
    window_start = Column(DateTime); window_end = Column(DateTime)
    score = Column(Float); mc_spread = Column(Float, nullable=True)
    alarm_1pct = Column(Boolean); alarm_01pct = Column(Boolean)
    n_local = Column(Integer); n_global = Column(Integer); unk_rate_local = Column(Float); unk_rate_global = Column(Float)
    n_input_lines = Column(Integer); n_malformed = Column(Integer); input_sha1 = Column(String(40)); model_run = Column(String(64))
    latency_ms = Column(Float)
    feedback = relationship("Feedback", back_populates="prediction", uselist=False)


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), unique=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    correct = Column(Boolean); note = Column(Text, nullable=True)
    prediction = relationship("Prediction", back_populates="feedback")
