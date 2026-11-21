"""影子试跑路由：发起、重试、封存与对照查询。派工仍只依据正式等级。"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from . import trial_service
from .models import TrialRun, User
from .schemas import TrialStartIn
from .security import current_user, get_db, require_roles

router = APIRouter(prefix="/api/trials", tags=["影子试跑"])


@router.get("")
def list_trials(db: Session = Depends(get_db), user: User = Depends(current_user)):
    return trial_service.list_trials(db)


@router.get("/{trial_id}")
def get_trial(trial_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    trial = db.get(TrialRun, trial_id)
    if not trial:
        raise HTTPException(404, "试跑不存在")
    return trial_service.trial_dict(db, trial)


@router.post("", status_code=201)
def start_trial(payload: TrialStartIn, db: Session = Depends(get_db),
                user: User = Depends(require_roles("operator", "supervisor"))):
    trial = trial_service.start_trial(db, user, payload.note)
    return trial_service.trial_dict(db, trial)


@router.post("/{trial_id}/retry")
def retry_trial(trial_id: int, case_id: int | None = Query(default=None),
                db: Session = Depends(get_db),
                user: User = Depends(require_roles("operator", "supervisor"))):
    return trial_service.retry_trial(db, user, trial_id, case_id)


@router.post("/{trial_id}/seal")
def seal_trial(trial_id: int, db: Session = Depends(get_db),
               user: User = Depends(require_roles("supervisor"))):
    trial = trial_service.seal_trial(db, user, trial_id)
    return trial_service.trial_dict(db, trial)
