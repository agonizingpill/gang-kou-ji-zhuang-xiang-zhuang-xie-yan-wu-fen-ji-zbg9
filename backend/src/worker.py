import json, os, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from redis import Redis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select
from .database import Base, SessionLocal, engine
from .models import AnalysisJob, DelayCase, TrialResult, TrialRun

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
SHARED_DIR = Path(os.getenv("SHARED_DIR", "/shared"))
queue = Redis.from_url(REDIS_URL, decode_responses=True)

OFFICIAL_QUEUE = "container_delay_triage_jobs"
TRIAL_QUEUE = "container_delay_trial_jobs"

BASE_FEATURE_KEYS = ["container_count", "avg_gate_dwell_min", "max_gate_dwell_min", "reefer_ratio", "hazmat_ratio", "customs_open_count", "gate_event_count", "dry_count", "reefer_count", "hazmat_count", "port_call_hour", "window_minutes"]
CROSS_FEATURE_KEYS = ["dwell_x_reefer", "gate_x_count"]
FEATURE_KEYS = BASE_FEATURE_KEYS + CROSS_FEATURE_KEYS

def vector(f, keys):
    return np.array([[f[k] for k in keys]], dtype=float)

TRAIN_ROWS = [
    ([4, 8, 12, .0, .0, 0, 4, 4, 0, 0, 6, 480], "normal"),
    ([6, 16, 22, .17, .0, 0, 6, 5, 1, 0, 8, 420], "watch"),
    ([8, 28, 40, .25, .12, 0, 8, 5, 2, 1, 10, 360], "major"),
    ([9, 42, 66, .33, .22, 1, 9, 4, 3, 2, 13, 300], "critical"),
    ([5, 10, 18, .2, .0, 0, 5, 4, 1, 0, 7, 520], "normal"),
    ([7, 22, 31, .29, .0, 0, 7, 4, 2, 0, 9, 390], "watch"),
    ([10, 35, 51, .3, .1, 0, 10, 6, 3, 1, 11, 330], "major"),
    ([12, 50, 78, .42, .17, 0, 12, 6, 5, 2, 14, 280], "critical"),
    ([3, 7, 10, .0, .0, 0, 3, 3, 0, 0, 5, 600], "normal"),
    ([8, 25, 36, .38, .0, 0, 8, 5, 3, 0, 12, 360], "major"),
    ([5, 18, 24, .2, .2, 0, 5, 3, 1, 1, 9, 440], "watch"),
    ([11, 45, 70, .45, .25, 0, 11, 4, 5, 2, 15, 260], "critical"),
]

def train_model(keys, with_cross):
    x, y = [], []
    for base, label in TRAIN_ROWS:
        f = dict(zip(BASE_FEATURE_KEYS, base))
        if with_cross:
            f["dwell_x_reefer"] = f["avg_gate_dwell_min"] * f["reefer_ratio"]
            f["gate_x_count"] = f["gate_event_count"] * f["container_count"]
        x.append([f[k] for k in keys]); y.append(label)
    model = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=500, random_state=42, multi_class="auto"))])
    model.fit(x, y)
    return model

# 现行模型：12 项基础特征 + 2 项手工交叉特征
MODEL = train_model(FEATURE_KEYS, with_cross=True)
# 候选模型：同一份训练样本、同一管线，仅去掉交叉特征
CANDIDATE_MODEL = train_model(BASE_FEATURE_KEYS, with_cross=False)

def predict(model, keys, f, formula):
    x = vector(f, keys)
    label = model.predict(x)[0]
    probs = model.predict_proba(x)[0]
    classes = model.named_steps["lr"].classes_
    return {"label": label, "confidence": round(float(max(probs)), 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
            "feature_keys": keys, "features": f, "formula": formula}

OFFICIAL_FORMULA = "手工交叉特征 = avg_gate_dwell_min×reefer_ratio, gate_event_count×container_count；StandardScaler + LogisticRegression(random_state=42)"
CANDIDATE_FORMULA = "候选模型已去除手工交叉特征，仅保留 12 项基础特征；StandardScaler + LogisticRegression(random_state=42)"

def run_official_job(db, job, case):
    f = dict(job.input_summary or {})
    f["dwell_x_reefer"] = round(f.get("avg_gate_dwell_min", 0) * f.get("reefer_ratio", 0), 3)
    f["gate_x_count"] = f.get("gate_event_count", 0) * f.get("container_count", 0)
    result = predict(MODEL, FEATURE_KEYS, f, OFFICIAL_FORMULA)
    job.result = result; job.status = "succeeded"; job.finished_at = datetime.now(timezone.utc)
    case.model_output = result; case.final_level = result["label"]; case.status = "classified"; case.exception_reason = None

def run_shadow_job(db, job, case):
    # 候选模型只吃同一次正式分级留下的输入快照，不重新取数
    f = dict(job.input_summary or {})
    result = predict(CANDIDATE_MODEL, BASE_FEATURE_KEYS, f, CANDIDATE_FORMULA)
    job.result = result; job.finished_at = datetime.now(timezone.utc)
    trial = db.get(TrialRun, job.trial_run_id) if job.trial_run_id else None
    row = db.execute(
        select(TrialResult).where(TrialResult.trial_run_id == job.trial_run_id, TrialResult.case_id == job.case_id)
    ).scalar_one_or_none()
    # 封存后才跑完的推理一律不计入对照，只在任务上留痕
    if trial is None or trial.status == "sealed" or row is None or row.status == "missing_at_seal":
        job.status = "excluded_late"; job.error = "试跑已封存，晚到的候选模型推理不计入对照"
        if row is not None:
            row.excluded_late = True
        return
    job.status = "succeeded"
    row.status = "succeeded"; row.candidate_level = result["label"]
    row.candidate_confidence = result["confidence"]; row.excluded_late = False

def run_job(job_id):
    db = SessionLocal()
    job = db.get(AnalysisJob, int(job_id))
    if not job:
        db.close(); return
    case = db.get(DelayCase, job.case_id)
    if case is None:
        job.status = "failed"; job.error = "关联延误案件不存在"
        job.finished_at = datetime.now(timezone.utc)
        db.commit(); db.close(); return
    try:
        job.status = "running"; db.commit()
        if job.job_scope == "shadow":
            run_shadow_job(db, job, case)
        else:
            run_official_job(db, job, case)
        db.commit()
    except Exception as exc:
        db.rollback()
        job = db.get(AnalysisJob, int(job_id))
        job.status = "failed"; job.error = str(exc); job.finished_at = datetime.now(timezone.utc)
        if job.job_scope == "shadow" and job.trial_run_id is not None:
            row = db.execute(
                select(TrialResult).where(TrialResult.trial_run_id == job.trial_run_id, TrialResult.case_id == job.case_id)
            ).scalar_one_or_none()
            if row is not None and row.status != "missing_at_seal":
                row.status = "failed"
        db.commit()
    finally:
        db.close()

def main():
    Base.metadata.create_all(engine)
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        Path(SHARED_DIR / "worker-heartbeat").touch()
        # 正式分级队列优先，试跑推理不挤占正式派工链路
        item = queue.blpop(OFFICIAL_QUEUE, timeout=1) or queue.blpop(TRIAL_QUEUE, timeout=2)
        if item:
            run_job(item[1])
        time.sleep(1)

if __name__ == "__main__":
    main()
