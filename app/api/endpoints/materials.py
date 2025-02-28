from fastapi import APIRouter, Depends, HTTPException, status, Query, File, UploadFile, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import logging
from pathlib import Path
import shutil

from app.db.database import get_db
from app.db import crud
from app.schemas import material as schemas
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/", response_model=List[schemas.Material])
def read_materials(
    skip: int = 0, 
    limit: int = 100, 
    search: Optional[str] = None,
    tag_ids: List[int] = Query(None),
    db: Session = Depends(get_db)
):
    """获取素材列表，支持搜索和标签过滤"""
    materials = crud.get_materials(db, skip=skip, limit=limit, search=search, tag_ids=tag_ids)
    
    # 为每个素材添加标签信息
    result = []
    for material in materials:
        material_dict = schemas.MaterialInDB.from_orm(material).dict()
        material_dict["tags"] = crud.get_material_tags(db, material.id)
        result.append(schemas.Material(**material_dict))
    
    return result

@router.get("/{material_id}", response_model=schemas.MaterialWithTags)
def read_material(material_id: int, db: Session = Depends(get_db)):
    """获取单个素材详情"""
    material = crud.get_material(db, material_id)
    if not material:
        raise HTTPException(status_code=404, detail="素材不存在")
    
    # 获取素材标签
    tags = crud.get_material_tags(db, material_id)
    
    # 获取素材所属项目
    projects = [project.id for project in material.projects]
    
    # 构建响应
    material_dict = schemas.MaterialInDB.from_orm(material).dict()
    material_dict["tags"] = tags
    material_dict["projects"] = projects
    
    return schemas.MaterialWithTags(**material_dict)

@router.get("/{material_id}/image")
def get_material_image(material_id: int, db: Session = Depends(get_db)):
    """获取素材图像文件"""
    material = crud.get_material(db, material_id)
    if not material:
        raise HTTPException(status_code=404, detail="素材不存在")
    
    if not os.path.exists(material.frame_path):
        raise HTTPException(status_code=404, detail="素材图像文件不存在")
    
    return FileResponse(material.frame_path)

@router.put("/{material_id}", response_model=schemas.Material)
def update_material(
    material_id: int, 
    material_update: schemas.MaterialUpdate, 
    db: Session = Depends(get_db)
):
    """更新素材信息"""
    db_material = crud.get_material(db, material_id)
    if not db_material:
        raise HTTPException(status_code=404, detail="素材不存在")
    
    updated_material = crud.update_material(db, material_id, material_update)
    
    # 获取素材标签
    tags = crud.get_material_tags(db, material_id)
    
    material_dict = schemas.MaterialInDB.from_orm(updated_material).dict()
    material_dict["tags"] = tags
    
    return schemas.Material(**material_dict)

@router.delete("/{material_id}")
def delete_material(material_id: int, db: Session = Depends(get_db)):
    """删除素材"""
    material = crud.get_material(db, material_id)
    if not material:
        raise HTTPException(status_code=404, detail="素材不存在")
    
    # 删除素材文件
    try:
        if os.path.exists(material.frame_path):
            os.remove(material.frame_path)
            
            # 如果文件夹为空，也删除文件夹
            folder = os.path.dirname(material.frame_path)
            if os.path.exists(folder) and not os.listdir(folder):
                os.rmdir(folder)
    except Exception as e:
        logger.error(f"删除文件失败: {e}")
    
    # 删除数据库记录
    if not crud.delete_material(db, material_id):
        raise HTTPException(status_code=500, detail="删除素材失败")
    
    return {"success": True, "message": "素材已删除"}

@router.post("/{material_id}/tags")
def add_material_tags(
    material_id: int, 
    tag_ids: List[int], 
    db: Session = Depends(get_db)
):
    """为素材添加标签"""
    material = crud.get_material(db, material_id)
    if not material:
        raise HTTPException(status_code=404, detail="素材不存在")
    
    if not crud.add_material_tags(db, material_id, tag_ids):
        raise HTTPException(status_code=500, detail="添加标签失败")
    
    return {"success": True, "message": "标签已添加"}

@router.delete("/{material_id}/tags/{tag_id}")
def remove_material_tag(
    material_id: int, 
    tag_id: int, 
    db: Session = Depends(get_db)
):
    """移除素材的标签"""
    material = crud.get_material(db, material_id)
    if not material:
        raise HTTPException(status_code=404, detail="素材不存在")
    
    tag = crud.get_tag(db, tag_id)
    if not tag:
        raise HTTPException(status_code=404, detail="标签不存在")
    
    if not crud.remove_material_tag(db, material_id, tag_id):
        raise HTTPException(status_code=500, detail="移除标签失败")
    
    return {"success": True, "message": "标签已移除"}