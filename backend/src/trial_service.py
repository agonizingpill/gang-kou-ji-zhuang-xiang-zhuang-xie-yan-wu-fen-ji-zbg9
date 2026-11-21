"""影子试跑：用去交叉特征的候选模型复算正式分级，全程不影响派工。"""

from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import redis_client
from .models import AnalysisJob, AuditLog, DelayCase, TrialResult, TrialRun

TRIAL_QUEUE = "container_delay_trial_jobs"
LEVEL_ORDER = {"critical": 4, "major": 3, "watch": 2, "normal": 1}


def audit(db: Session, user, action: str, trial_id: int, detail: str = ""):
    db.add(AuditLog(actor=user.username, role=user.role, action=action,
                    entity="trial_run", entity_id=trial_id, detail=detail))


def _latest_official_job(db: Session, case_id: int):
    return db.execute(
        select(AnalysisJob)
        .where(AnalysisJob.case_id == case_id, AnalysisJob.job_scope == "official",
               AnalysisJob.status == "succeeded", AnalysisJob.result.is_not(None))
        .order_by(AnalysisJob.id.desc())
    ).scalars().first()


def start_trial(db: Session, user, note: str):
    existing = db.execute(select(TrialRun).where(TrialRun.status == "running")).scalar_one_or_none()
    if existing:
        raise HTTPException(409, f"已有进行中的试跑 #{existing.id}，请先封存后再发起")
    cases = db.execute(select(DelayCase).order_by(DelayCase.id)).scalars().all()
    picked = []
    for case in cases:
        job = _latest_official_job(db, case.id)
        if job and job.input_summary:
            picked.append((case, job))
    if not picked:
        raise HTTPException(409, "暂无可试跑案件：需要至少一件已由正式模型成功分级的延误案件")

    trial = TrialRun(status="running", created_by=user.username, note=note or "去除交叉特征候选模型试跑")
    db.add(trial); db.flush()
    for case, official_job in picked:
        official_label = official_job.result.get("label") if official_job.result else case.final_level
        # 输入直接复制正式分级那一次的快照，候选与现行模型同一次输入
        shadow_job = AnalysisJob(
            case_id=case.id, job_type="delay_triage", job_scope="shadow", trial_run_id=trial.id,
            status="queued", input_summary=dict(official_job.input_summary),
        )
        db.add(shadow_job); db.flush()
        db.add(TrialResult(
            trial_run_id=trial.id, case_id=case.id,
            official_job_id=official_job.id, shadow_job_id=shadow_job.id,
            status="queued", input_snapshot=dict(official_job.input_summary),
            official_level=official_label,
            official_confidence=official_job.result.get("confidence") if official_job.result else None,
        ))
    db.commit()
    rows = db.execute(
        select(AnalysisJob.id).where(AnalysisJob.trial_run_id == trial.id, AnalysisJob.job_scope == "shadow")
    ).scalars().all()
    for job_id in rows:
        redis_client.rpush(TRIAL_QUEUE, str(job_id))
    audit(db, user, "trial_started", trial.id, f"纳入 {len(picked)} 件已正式分级案件，输入沿用正式分级快照")
    db.commit()
    return trial


def _get_running_trial(db: Session, trial_id: int) -> TrialRun:
    trial = db.get(TrialRun, trial_id)
    if not trial:
        raise HTTPException(404, "试跑不存在")
    if trial.status != "running":
        raise HTTPException(409, "试跑已封存，不能再重试或追加推理")
    return trial


def retry_trial(db: Session, user, trial_id: int, case_id: int | None):
    """失败重试：只重置原来的影子任务并原样重排队，不换输入、不新增计数。"""
    _get_running_trial(db, trial_id)
    stmt = select(AnalysisJob).where(
        AnalysisJob.trial_run_id == trial_id, AnalysisJob.job_scope == "shadow",
        AnalysisJob.status == "failed",
    )
    if case_id is not None:
        if not db.get(DelayCase, case_id):
            raise HTTPException(404, "延误案件不存在")
        stmt = stmt.where(AnalysisJob.case_id == case_id)
    jobs = db.execute(stmt).scalars().all()
    if not jobs:
        raise HTTPException(409, "没有可重试的失败影子任务")
    for job in jobs:
        # 同一 job 行原地重试：input_summary 保持不变，同一次分级只计这一条
        job.status = "queued"; job.error = None; job.finished_at = None
        row = db.execute(
            select(TrialResult).where(TrialResult.trial_run_id == trial_id, TrialResult.case_id == job.case_id)
        ).scalar_one()
        row.status = "queued"; row.candidate_level = None; row.candidate_confidence = None
    db.commit()
    for job in jobs:
        redis_client.rpush(TRIAL_QUEUE, str(job.id))
    audit(db, user, "trial_retried", trial_id,
          f"原任务原地重试 {len(jobs)} 条，输入保持不变：{[j.id for j in jobs]}")
    db.commit()
    return {"retried_job_ids": [j.id for j in jobs], "count": len(jobs)}


def seal_trial(db: Session, user, trial_id: int):
    trial = _get_running_trial(db, trial_id)
    pending = db.execute(
        select(TrialResult).where(TrialResult.trial_run_id == trial_id,
                                  TrialResult.status.in_(("queued", "running", "failed")))
    ).scalars().all()
    for row in pending:
        # 封存时刻仍缺结果（在途或失败）的案件固定为缺结果，晚到推理不再计入
        row.status = "missing_at_seal"
    trial.status = "sealed"; trial.sealed_by = user.username
    trial.sealed_at = datetime.now(timezone.utc)
    audit(db, user, "trial_sealed", trial.id,
          f"封存时缺结果案件 {len(pending)} 件，晚到推理不再计入")
    db.commit()
    return trial


def result_dict(row: TrialResult, case: DelayCase):
    downgraded = (
        row.candidate_level is not None
        and LEVEL_ORDER.get(row.candidate_level, 0) < LEVEL_ORDER.get(row.official_level, 0)
    )
    return {
        "case_id": row.case_id, "case_no": case.case_no,
        "vessel": case.vessel.call_no if case.vessel else "",
        "vessel_name": case.vessel.vessel_name if case.vessel else "",
        "status": row.status, "excluded_late": row.excluded_late,
        "official_level": row.official_level, "candidate_level": row.candidate_level,
        "official_confidence": row.official_confidence, "candidate_confidence": row.candidate_confidence,
        "downgraded": downgraded,
        "shadow_job_id": row.shadow_job_id,
    }


def trial_dict(db: Session, trial: TrialRun):
    rows = db.execute(
        select(TrialResult, DelayCase).join(DelayCase, DelayCase.id == TrialResult.case_id)
        .where(TrialResult.trial_run_id == trial.id).order_by(TrialResult.id)
    ).all()
    items = [result_dict(row, case) for row, case in rows]
    counts = {"total": len(items), "succeeded": 0, "failed": 0, "missing_at_seal": 0, "downgraded": 0}
    for item in items:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
        if item["downgraded"]:
            counts["downgraded"] += 1
    return {
        "id": trial.id, "status": trial.status, "note": trial.note,
        "created_by": trial.created_by, "sealed_by": trial.sealed_by,
        "created_at": trial.created_at.isoformat(),
        "sealed_at": trial.sealed_at.isoformat() if trial.sealed_at else None,
        "counts": counts, "results": items,
    }


def list_trials(db: Session):
    trials = db.execute(select(TrialRun).order_by(TrialRun.id.desc())).scalars().all()
    return [trial_dict(db, t) for t in trials]
