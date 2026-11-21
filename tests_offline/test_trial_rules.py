"""离线业务规则测试：候选模型、封存排除、重试幂等、派工隔离。

运行：python3 -I tests_offline/test_trial_rules.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

import stubs_bootstrap as stubs  # noqa: E402

# 用 stub 顶替真实第三方库
for name in ("numpy", "redis", "sklearn", "sklearn.linear_model", "sklearn.pipeline",
             "sklearn.preprocessing", "sqlalchemy", "sqlalchemy.orm", "fastapi"):
    sys.modules.setdefault(name, stubs)
sys.modules["numpy"] = stubs
sys.modules["redis"] = stubs
sys.modules["sqlalchemy"] = stubs
sys.modules["fastapi"] = stubs
import types  # noqa: E402

orm = types.ModuleType("sqlalchemy.orm")
orm.Session = stubs.Session

class _Mapped:
    def __class_getitem__(cls, item):
        return None

orm.Mapped = _Mapped
orm.mapped_column = stubs.mapped_column
orm.relationship = stubs.relationship
orm.DeclarativeBase = stubs.DeclarativeBase
orm.sessionmaker = stubs.sessionmaker
sys.modules["sqlalchemy.orm"] = orm

linear = types.ModuleType("sklearn.linear_model")
linear.LogisticRegression = stubs.LogisticRegression
sys.modules["sklearn.linear_model"] = linear
pipeline_mod = types.ModuleType("sklearn.pipeline")
pipeline_mod.Pipeline = stubs.Pipeline
sys.modules["sklearn.pipeline"] = pipeline_mod
prep = types.ModuleType("sklearn.preprocessing")
prep.StandardScaler = stubs.StandardScaler
sys.modules["sklearn.preprocessing"] = prep
sk = types.ModuleType("sklearn")
sys.modules["sklearn"] = sk

from src.database import Base, SessionLocal, engine  # noqa: E402
from src.models import AnalysisJob, AuditLog, DelayCase, TrialResult, TrialRun  # noqa: E402
from src import worker, trial_service  # noqa: E402

PASS, FAIL = 0, 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} {detail}")

def expect_http(status, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except stubs.HTTPException as exc:
        return exc.status_code == status, exc.detail
    return False, "未抛出 HTTPException"

class Actor:
    def __init__(self, username, role):
        self.username = username
        self.role = role

operator = Actor("operator", "operator")
supervisor = Actor("supervisor", "supervisor")

def features(container_count, dwell):
    f = {
        "container_count": container_count, "avg_gate_dwell_min": dwell,
        "max_gate_dwell_min": dwell + 4, "reefer_ratio": 0.25, "hazmat_ratio": 0.1,
        "customs_open_count": 0, "gate_event_count": container_count,
        "dry_count": container_count - 2, "reefer_count": 1, "hazmat_count": 1,
        "port_call_hour": 9, "window_minutes": 360,
    }
    return f

# 现行模型按 avg dwell 分档；候选模型（去交叉）按 container_count 分档
SCENARIOS = [
    # case_no, 箱量, 平均停留 → 现行(按停留)/候选(按箱量)
    ("CASE-D-001", 4, 45, "critical", "normal"),   # 判轻：critical → normal
    ("CASE-D-002", 6, 30, "major", "watch"),        # 判轻：major → watch
    ("CASE-D-003", 8, 18, "watch", "major"),        # 判重
    ("CASE-D-004", 10, 20, "watch", "critical"),    # 判重
]

def reset_db():
    engine.tables.clear()
    engine.next_ids.clear()

def seed_official_cases():
    db = SessionLocal()
    for case_no, count, dwell, official_label, candidate_label in SCENARIOS:
        case = DelayCase(case_no=case_no, vessel_id=1, status="classified",
                         final_level=official_label)
        db.add(case); db.flush()
        job = AnalysisJob(case_id=case.id, job_scope="official", status="succeeded",
                          input_summary=features(count, dwell),
                          result={"label": official_label, "confidence": 0.71})
        db.add(job); db.flush()
    db.commit()
    return db

print("== 1. 候选模型确实去掉交叉特征，训练数据与管线不变 ==")
check("现行模型 14 维（12 基础 + 2 交叉）", worker.FEATURE_KEYS[-2:] == ["dwell_x_reefer", "gate_x_count"]
      and len(worker.FEATURE_KEYS) == 14)
check("候选模型仅 12 项基础特征", worker.BASE_FEATURE_KEYS[-1] == "window_minutes"
      and len(worker.BASE_FEATURE_KEYS) == 12
      and not any(k.endswith(("_x_",)) for k in worker.BASE_FEATURE_KEYS))
check("两份模型同训练样本、同 random_state=42", worker.MODEL.named_steps["lr"].random_state == 42
      and worker.CANDIDATE_MODEL.named_steps["lr"].random_state == 42)
check("候选公式注明已去交叉特征", "去除手工交叉特征" in worker.CANDIDATE_FORMULA)

print("== 2. 发起试跑：同一次正式分级输入，影子任务不触碰案件正式等级 ==")
reset_db()
db = seed_official_cases()
case_before = {c.case_no: (c.final_level, c.status) for c in engine.table(DelayCase).values()}
trial = trial_service.start_trial(db, operator, "")
db.commit()
check("试跑状态 running", trial.status == "running")
rows = list(engine.table(TrialResult).values())
shadow_jobs = [j for j in engine.table(AnalysisJob).values() if j.job_scope == "shadow"]
check("每个已分级案件一条对照，每个对照一个影子任务",
      len(rows) == 4 and len(shadow_jobs) == 4)
for case_no, count, dwell, official_label, candidate_label in SCENARIOS:
    case = next(c for c in engine.table(DelayCase).values() if c.case_no == case_no)
    row = next(r for r in rows if r.case_id == case.id)
    check(f"{case_no} 影子输入与正式输入完全一致",
          row.input_snapshot == features(count, dwell)
          and row.shadow_job_id is not None
          and engine.table(AnalysisJob)[row.shadow_job_id].input_summary == features(count, dwell))
    check(f"{case_no} 现行等级取自正式分级", row.official_level == official_label)
check("派工依据的正式等级与状态完全不变",
      {c.case_no: (c.final_level, c.status) for c in engine.table(DelayCase).values()} == case_before)
check("影子任务全部入独立试跑队列",
      stubs.Redis._instances[worker.REDIS_URL].queues[worker.TRIAL_QUEUE] == [str(j.id) for j in shadow_jobs])
check("发起动作写审计", any(a.action == "trial_started" for a in engine.table(AuditLog).values()))
ok, detail = expect_http(409, trial_service.start_trial, SessionLocal(), operator, "")
check("同时只能有一个进行中的试跑", ok, detail)

print("== 3. worker 跑影子任务：候选等级只写对照，不回写案件 ==")
queued = list(stubs.Redis._instances[worker.REDIS_URL].queues[worker.TRIAL_QUEUE])
for job_id in queued:
    worker.run_job(int(job_id))
for case_no, count, dwell, official_label, candidate_label in SCENARIOS:
    case = next(c for c in engine.table(DelayCase).values() if c.case_no == case_no)
    row = next(r for r in engine.table(TrialResult).values() if r.case_id == case.id)
    check(f"{case_no} 候选等级={candidate_label}（现行 {official_label}）",
          row.status == "succeeded" and row.candidate_level == candidate_label,
          f"got {row.candidate_level}")
    check(f"{case_no} 案件 final_level 仍是正式等级", case.final_level == official_label)
downgraded = [r for r in engine.table(TrialResult).values()
              if trial_service.LEVEL_ORDER[r.candidate_level] < trial_service.LEVEL_ORDER[r.official_level]]
check("判轻（降级）案件识别为 2 件",
      sorted(engine.table(DelayCase)[r.case_id].case_no for r in downgraded) == ["CASE-D-001", "CASE-D-002"])
check("影子任务成功且结果不含交叉特征",
      all(j.status == "succeeded" for j in shadow_jobs)
      and all("dwell_x_reefer" not in j.result["features"] for j in shadow_jobs)
      and all(j.result["feature_keys"] == worker.BASE_FEATURE_KEYS for j in shadow_jobs))

print("== 4. 正式分级走原链路：仍含交叉特征并回写案件 ==")
fresh_case = DelayCase(case_no="CASE-OFF-01", vessel_id=1, status="queued")
db = SessionLocal(); db.add(fresh_case); db.flush()
off_job = AnalysisJob(case_id=fresh_case.id, job_scope="official", status="queued",
                      input_summary=features(6, 45))
db.add(off_job); db.commit()
worker.run_job(off_job.id)
check("正式任务成功并回写案件等级",
      off_job.status == "succeeded" and fresh_case.status == "classified"
      and fresh_case.final_level == off_job.result["label"] == "critical")
check("正式结果仍含两个交叉特征",
      set(off_job.result["feature_keys"][-2:]) == {"dwell_x_reefer", "gate_x_count"})

print("== 5. 失败重试：原地同一任务、不换输入、不重复计数 ==")
retry_trial = TrialRun(status="running", created_by="operator"); db.add(retry_trial); db.flush()
retry_case = DelayCase(case_no="CASE-R-01", vessel_id=1, status="classified", final_level="major")
db.add(retry_case); db.flush()
base_input = features(6, 28)
base_job = AnalysisJob(case_id=retry_case.id, job_scope="official", status="succeeded",
                       input_summary=base_input, result={"label": "major", "confidence": 0.7})
db.add(base_job); db.flush()
shadow = AnalysisJob(case_id=retry_case.id, job_scope="shadow", trial_run_id=retry_trial.id,
                     status="queued", input_summary=dict(base_input))
db.add(shadow); db.flush()
trow = TrialResult(trial_run_id=retry_trial.id, case_id=retry_case.id,
                   official_job_id=base_job.id, shadow_job_id=shadow.id,
                   status="queued", input_snapshot=dict(base_input), official_level="major")
db.add(trow); db.commit()
shadow_id = shadow.id
# 模拟 worker 失败
shadow.status = "failed"; shadow.error = "boom"; trow.status = "failed"; db.commit()
out = trial_service.retry_trial(SessionLocal(), operator, retry_trial.id, None)
check("重试返回原任务 id，数量 1", out == {"retried_job_ids": [shadow_id], "count": 1})
check("仍是同一个 job 行（无新任务、无新对照）",
      len([j for j in engine.table(AnalysisJob).values() if j.trial_run_id == retry_trial.id]) == 1
      and len([r for r in engine.table(TrialResult).values() if r.trial_run_id == retry_trial.id]) == 1)
shadow = engine.table(AnalysisJob)[shadow_id]
check("重试后状态回 queued，错误清空", shadow.status == "queued" and shadow.error is None)
check("输入快照原样保留", shadow.input_summary == base_input and trow.input_snapshot == base_input)
check("失败任务重新进入试跑队列",
      str(shadow_id) in stubs.Redis._instances[worker.REDIS_URL].queues[worker.TRIAL_QUEUE])
ok, detail = expect_http(409, trial_service.retry_trial, SessionLocal(), operator, retry_trial.id, None)
check("没有失败任务时重试被拒", ok, detail)
# 重试成功
worker.run_job(shadow_id)
check("重试后影子任务成功并写入候选等级",
      shadow.status == "succeeded" and trow.status == "succeeded" and trow.candidate_level == "watch")
check("重试写审计", any(a.action == "trial_retried" for a in engine.table(AuditLog).values()))
trial_service.seal_trial(SessionLocal(), supervisor, retry_trial.id)

print("== 6. 封存：在途/失败固定为缺结果，晚到推理不计入 ==")
reset_db()
seed_official_cases()
seal_trial = trial_service.start_trial(SessionLocal(), operator, "")
seal_rows = [r for r in engine.table(TrialResult).values() if r.trial_run_id == seal_trial.id]
check("重置后新试跑覆盖 4 件正式分级案件", len(seal_rows) == 4)
# 先让两件跑完候选推理，再留一件在途、一件失败
for row in seal_rows[2:]:
    worker.run_job(row.shadow_job_id)
running_row = seal_rows[0]; failed_row = seal_rows[1]
running_job = engine.table(AnalysisJob)[running_row.shadow_job_id]
failed_job = engine.table(AnalysisJob)[failed_row.shadow_job_id]
running_job.status = "running"; running_row.status = "running"
failed_job.status = "failed"; failed_row.status = "failed"
db.commit()
sealed = trial_service.seal_trial(SessionLocal(), supervisor, seal_trial.id)
check("封存状态与封存人", sealed.status == "sealed" and sealed.sealed_by == "supervisor")
check("在途与失败均标记为封存时缺结果",
      running_row.status == "missing_at_seal" and failed_row.status == "missing_at_seal")
check("缺结果计数 = 2", trial_service.trial_dict(SessionLocal(), sealed)["counts"]["missing_at_seal"] == 2)
ok, detail = expect_http(409, trial_service.retry_trial, SessionLocal(), operator, seal_trial.id, None)
check("封存后禁止重试", ok, detail)
# 晚到的在途任务此时才跑完
late_job = engine.table(AnalysisJob)[running_row.shadow_job_id]
worker.run_job(late_job.id)
check("晚到任务标记 excluded_late，对照候选等级仍为空",
      late_job.status == "excluded_late" and running_row.candidate_level is None
      and running_row.status == "missing_at_seal" and running_row.excluded_late is True)
check("晚到结果不改变封存计数",
      trial_service.trial_dict(SessionLocal(), sealed)["counts"]["missing_at_seal"] == 2)
check("封存写审计", any(a.action == "trial_sealed" for a in engine.table(AuditLog).values()))
# 封存后再发起新试跑不受影响
new_trial = trial_service.start_trial(SessionLocal(), operator, "")
check("封存后可发起新一轮试跑", new_trial.status == "running" and new_trial.id != seal_trial.id)

print("== 7. 无成功正式分级时拒绝试跑 ==")
reset_db()
empty_db = SessionLocal()
ok, detail = expect_http(409, trial_service.start_trial, empty_db, operator, "")
check("没有可试跑案件时返回 409", ok, detail)

print(f"\n结果：{PASS} 通过，{FAIL} 失败")
sys.exit(1 if FAIL else 0)
