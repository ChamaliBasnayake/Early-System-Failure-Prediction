from __future__ import annotations

import hashlib
import time

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.backend import settings
from app.backend.db.models import Feedback, Prediction
from app.backend.db.session import get_db
from app.backend.schemas import FeedbackRequest, PredictRequest, PredictResponse, UnitResult
from app.backend.services.featurize import Featurizer
from app.backend.services.inference import ModelService

router = APIRouter(prefix="/api")
_feat: Featurizer | None = None; _svc: ModelService | None = None
SCORE_MEANING = ("The score is the model's ranking signal for 'a kernel-level failure starts on this midplane within the lead time'. "
                 "It is NOT a calibrated probability. 'Alarm at 1 %' means the score exceeds the validation threshold that produced about "
                 "14 false alarms per midplane per day; 'Alarm at 0.1 %' corresponds to about 1.4. The Monte-Carlo spread shows how much "
                 "the score moves when dropout is active; a large spread means an unstable score, not a confidence interval.")


def services():
    global _feat, _svc
    if _feat is None: _feat = Featurizer()
    if _svc is None: _svc = ModelService()
    return _feat, _svc


@router.get("/health")
def health():
    feat, svc = services()
    return {"status": "ok", "models": {N: m["run"] for N, m in svc.models.items()}, "known_texts": feat.n_known_texts, "db": settings.DATABASE_URL.split("://")[0]}


@router.get("/model")
def model_info():
    _, svc = services()
    return {"version": settings.MODEL_VERSION, "models": {N: {"run": m["run"], "n_params": m["n_params"], "thresholds": m["thresholds"]} for N, m in svc.models.items()},
            "window_min": settings.WINDOW_MIN, "limitations": settings.LIMITATIONS, "score_meaning": SCORE_MEANING}


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    feat, svc = services(); t0 = time.time()
    if not req.log_text or not req.log_text.strip(): raise HTTPException(400, "Empty input: paste the last hour of BGL log lines.")
    if req.log_text.count("\n") > settings.MAX_LINES: raise HTTPException(413, f"Input too large (> {settings.MAX_LINES} lines).")
    if req.lead_time not in (1, 5, 15, 30): raise HTTPException(400, "lead_time must be one of 1, 5, 15, 30 minutes.")
    parsed = feat.parse(req.log_text)
    if parsed.df.empty: raise HTTPException(422, f"No parsable BGL lines found ({parsed.n_malformed} malformed). Expected: '<tag> <epoch> <date> <node> <datetime> <node> RAS <component> <level> <message>'.")
    if parsed.n_malformed > 0.5 * parsed.n_input_lines: raise HTTPException(422, f"{parsed.n_malformed} of {parsed.n_input_lines} lines are not BGL-formatted; refusing to predict on mostly-invalid input.")
    units = [req.unit] if req.unit else parsed.units
    if req.unit and req.unit not in parsed.units: raise HTTPException(422, f"Midplane {req.unit} does not appear in the input. Detected: {parsed.units[:20]}")
    if len(units) > 200: raise HTTPException(413, "Too many midplanes in one request (max 200). Specify 'unit'.")
    sha = hashlib.sha1(req.log_text.encode("utf-8", "ignore")).hexdigest(); results = []; used_N = req.lead_time
    for u in units:
        win = feat.window(parsed, u); out = svc.score(win, req.lead_time, req.mc_samples); used_N = out["lead_time_used"]
        ood = win["unk_rate_global"] > 0.5 or win["n_global"] == 0
        alarm1 = bool(out["alarm_at_far"].get("0.01", False)); alarm01 = bool(out["alarm_at_far"].get("0.001", False))
        rec = ("Withhold this midplane from the next allocation and checkpoint running jobs" if alarm01 else
               "Raise attention on this midplane; consider checkpointing long jobs" if alarm1 else "No action")
        if ood: rec += " — input is far from the training distribution (high unknown-template rate); treat the score as unreliable"
        row = Prediction(unit=u, lead_time=req.lead_time, lead_time_used=out["lead_time_used"], window_start=win["window_start"].to_pydatetime(), window_end=win["window_end"].to_pydatetime(),
                         score=out["score"], mc_spread=out["mc_spread"], alarm_1pct=alarm1, alarm_01pct=alarm01, n_local=win["n_local"], n_global=win["n_global"],
                         unk_rate_local=win["unk_rate_local"], unk_rate_global=win["unk_rate_global"], n_input_lines=parsed.n_input_lines, n_malformed=parsed.n_malformed,
                         input_sha1=sha, model_run=out["run"], latency_ms=(time.time() - t0) * 1000)
        db.add(row); db.commit(); db.refresh(row)
        results.append(UnitResult(prediction_id=row.id, unit=u, score=out["score"], mc_spread=out["mc_spread"], mc_samples=out["mc_samples"], alarm_at_far=out["alarm_at_far"],
                                  thresholds=out["thresholds"], recommendation=rec, n_local=win["n_local"], n_global=win["n_global"], unk_rate_local=win["unk_rate_local"],
                                  unk_rate_global=win["unk_rate_global"], n_new_texts=win["n_new_texts"], out_of_distribution=ood, new_texts_embedded=out.get("new_texts_embedded", 0), encoder=out.get("encoder", ""),
                                  window_start=win["window_start"].isoformat(), window_end=win["window_end"].isoformat(), local_lines=win["lines"]))
    results.sort(key=lambda r: -r.score)
    return PredictResponse(model_run=svc.resolve(req.lead_time)[1]["run"], model_version=settings.MODEL_VERSION, lead_time_requested=req.lead_time,
                           lead_time_used=used_N, fallback_lead_time=used_N != req.lead_time, n_input_lines=parsed.n_input_lines, n_parsed=len(parsed.df), n_malformed=parsed.n_malformed,
                           units_detected=parsed.units, latency_ms=(time.time() - t0) * 1000, score_meaning=SCORE_MEANING, limitations=settings.LIMITATIONS, results=results)


EXPLAIN_MEANING = ("Integrated Gradients: how much each item (a run of one message type) and each context feature moved the model's logit, relative to an EMPTY window. "
                   "Positive values push towards 'failure', negative away. The values sum to the difference between this window's logit and the empty window's logit. "
                   "This shows what the model was sensitive to; it does not prove that a message caused the prediction, and correlated messages may share attribution.")


@router.post("/explain")
def explain(req: PredictRequest):
    feat, svc = services()
    if not req.unit: raise HTTPException(400, "Specify the midplane ('unit') to explain.")
    parsed = feat.parse(req.log_text or "")
    if parsed.df.empty: raise HTTPException(422, "No parsable BGL lines found.")
    if req.unit not in parsed.units: raise HTTPException(422, f"Midplane {req.unit} does not appear in the input.")
    t0 = time.time(); win = feat.window(parsed, req.unit); ex = svc.explain(win, req.lead_time)
    tpl_text = _template_texts(feat)
    for d in ex["items_local"] + ex["items_global"]:
        d["text"] = feat.text_of(d["masked_id"]) or tpl_text.get(d["template_index"], "UNK")
    return {"unit": req.unit, "meaning": EXPLAIN_MEANING, "latency_ms": (time.time() - t0) * 1000, "lines": win["lines"], **ex}


_tpl_cache: dict | None = None


def _template_texts(feat: Featurizer) -> dict:
    global _tpl_cache
    if _tpl_cache is None:
        t = pd.read_csv(settings.INTERIM / "templates.csv"); _tpl_cache = dict(zip(t["template_index"].astype(int), t["template"]))
    return _tpl_cache


@router.post("/feedback")
def feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    p = db.get(Prediction, req.prediction_id)
    if p is None: raise HTTPException(404, "prediction not found")
    fb = p.feedback or Feedback(prediction_id=p.id); fb.correct = req.correct; fb.note = req.note; db.add(fb); db.commit()
    return {"status": "stored", "prediction_id": p.id, "correct": req.correct}


@router.get("/history")
def history(limit: int = 50, db: Session = Depends(get_db)):
    rows = db.query(Prediction).order_by(Prediction.id.desc()).limit(min(limit, 500)).all()
    return [{"id": r.id, "created_at": r.created_at.isoformat() if r.created_at else None, "unit": r.unit, "lead_time": r.lead_time, "score": r.score, "mc_spread": r.mc_spread,
             "alarm_1pct": r.alarm_1pct, "alarm_01pct": r.alarm_01pct, "n_local": r.n_local, "n_global": r.n_global, "unk_rate_global": r.unk_rate_global,
             "model_run": r.model_run, "latency_ms": r.latency_ms, "feedback": (None if r.feedback is None else {"correct": r.feedback.correct, "note": r.feedback.note})} for r in rows]
