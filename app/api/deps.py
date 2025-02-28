from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.core.workflow import MaterialProcessWorkflow

def get_workflow(db: Session = Depends(get_db)) -> MaterialProcessWorkflow:
    """获取素材处理工作流实例"""
    return MaterialProcessWorkflow(db)