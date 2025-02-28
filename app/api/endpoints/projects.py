from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging

from app.db.database import get_db
from app.db import crud
from app.schemas import project as schemas

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/", response_model=List[schemas.Project])
def read_projects(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    """获取项目列表"""
    projects = crud.get_projects(db, skip=skip, limit=limit)
    
    # 为每个项目添加素材数量
    result = []
    for project in projects:
        # 获取项目素材数量
        material_count = db.query(crud.models.project_material).filter(
            crud.models.project_material.c.project_id == project.id
        ).count()
        
        project_dict = schemas.ProjectInDB.from_orm(project).dict()
        project_dict["material_count"] = material_count
        result.append(schemas.Project(**project_dict))
    
    return result

@router.get("/{project_id}", response_model=schemas.Project)
def read_project(project_id: int, db: Session = Depends(get_db)):
    """获取单个项目详情"""
    project = crud.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    # 获取项目素材数量
    material_count = db.query(crud.models.project_material).filter(
        crud.models.project_material.c.project_id == project.id
    ).count()
    
    project_dict = schemas.ProjectInDB.from_orm(project).dict()
    project_dict["material_count"] = material_count
    
    return schemas.Project(**project_dict)

@router.post("/", response_model=schemas.Project)
def create_project(project: schemas.ProjectCreate, db: Session = Depends(get_db)):
    """创建新项目"""
    # 检查项目名称是否已存在
    existing_project = crud.get_project_by_name(db, project.name)
    if existing_project:
        raise HTTPException(status_code=400, detail="项目名称已存在")
    
    db_project = crud.create_project(db, project)
    
    return schemas.Project(
        id=db_project.id,
        name=db_project.name,
        description=db_project.description,
        created_date=db_project.created_date,
        material_count=0
    )

@router.put("/{project_id}", response_model=schemas.Project)
def update_project(
    project_id: int, 
    project_update: schemas.ProjectUpdate, 
    db: Session = Depends(get_db)
):
    """更新项目信息"""
    db_project = crud.get_project(db, project_id)
    if not db_project:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    # 如果要更新名称，检查新名称是否已存在
    if project_update.name and project_update.name != db_project.name:
        existing_project = crud.get_project_by_name(db, project_update.name)
        if existing_project:
            raise HTTPException(status_code=400, detail="项目名称已存在")
    
    updated_project = crud.update_project(db, project_id, project_update)
    
    # 获取项目素材数量
    material_count = db.query(crud.models.project_material).filter(
        crud.models.project_material.c.project_id == project_id
    ).count()
    
    project_dict = schemas.ProjectInDB.from_orm(updated_project).dict()
    project_dict["material_count"] = material_count
    
    return schemas.Project(**project_dict)

@router.delete("/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    """删除项目"""
    project = crud.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    if not crud.delete_project(db, project_id):
        raise HTTPException(status_code=500, detail="删除项目失败")
    
    return {"success": True, "message": "项目已删除"}

@router.get("/{project_id}/materials")
def get_project_materials(
    project_id: int, 
    skip: int = 0, 
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """获取项目中的素材"""
    project = crud.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    materials = crud.get_project_materials(db, project_id, skip=skip, limit=limit)
    
    # 获取素材标签
    result = []
    for material in materials:
        tags = crud.get_material_tags(db, material.id)
        
        result.append({
            "id": material.id,
            "source_video": material.source_video,
            "frame_path": material.frame_path,
            "timestamp": material.timestamp,
            "tags": tags
        })
    
    return result

@router.post("/{project_id}/materials")
def add_materials_to_project(
    project_id: int, 
    data: schemas.ProjectMaterialAdd,
    db: Session = Depends(get_db)
):
    """将素材添加到项目"""
    project = crud.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    success_count = 0
    failed_ids = []
    
    for material_id in data.material_ids:
        # 检查素材是否存在
        material = crud.get_material(db, material_id)
        if not material:
            failed_ids.append(material_id)
            continue
        
        # 添加素材到项目
        if crud.add_material_to_project(db, project_id, material_id, data.notes):
            success_count += 1
        else:
            failed_ids.append(material_id)
    
    return {
        "success": True,
        "added_count": success_count,
        "failed_ids": failed_ids,
        "message": f"已添加 {success_count} 个素材到项目"
    }

@router.delete("/{project_id}/materials/{material_id}")
def remove_material_from_project(
    project_id: int, 
    material_id: int, 
    db: Session = Depends(get_db)
):
    """从项目中移除素材"""
    project = crud.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    material = crud.get_material(db, material_id)
    if not material:
        raise HTTPException(status_code=404, detail="素材不存在")
    
    if not crud.remove_material_from_project(db, project_id, material_id):
        raise HTTPException(status_code=500, detail="移除素材失败")
    
    return {"success": True, "message": "素材已从项目中移除"}