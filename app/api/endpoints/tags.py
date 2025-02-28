from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging

from app.db.database import get_db
from app.db import crud
from app.schemas import tag as schemas

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/", response_model=List[schemas.Tag])
def read_tags(
    skip: int = 0, 
    limit: int = 100, 
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """获取标签列表，可按类别过滤"""
    tags = crud.get_tags(db, skip=skip, limit=limit, category=category)
    
    # 为每个标签添加使用次数
    result = []
    for tag in tags:
        usage_count = db.query(crud.models.material_tag).filter(
            crud.models.material_tag.c.tag_id == tag.id
        ).count()
        
        tag_dict = schemas.TagInDB.from_orm(tag).dict()
        tag_dict["usage_count"] = usage_count
        result.append(schemas.Tag(**tag_dict))
    
    return result

@router.get("/categories", response_model=List[str])
def read_tag_categories(db: Session = Depends(get_db)):
    """获取所有标签类别"""
    return crud.get_tag_categories(db)

@router.get("/{tag_id}", response_model=schemas.Tag)
def read_tag(tag_id: int, db: Session = Depends(get_db)):
    """获取单个标签详情"""
    tag = crud.get_tag(db, tag_id)
    if not tag:
        raise HTTPException(status_code=404, detail="标签不存在")
    
    # 获取使用次数
    usage_count = db.query(crud.models.material_tag).filter(
        crud.models.material_tag.c.tag_id == tag.id
    ).count()
    
    tag_dict = schemas.TagInDB.from_orm(tag).dict()
    tag_dict["usage_count"] = usage_count
    
    return schemas.Tag(**tag_dict)

@router.post("/", response_model=schemas.Tag)
def create_tag(tag: schemas.TagCreate, db: Session = Depends(get_db)):
    """创建新标签"""
    # 检查标签是否已存在
    existing_tag = crud.get_tag_by_name(db, tag.name)
    if existing_tag:
        raise HTTPException(status_code=400, detail="标签已存在")
    
    db_tag = crud.create_tag(db, tag)
    
    return schemas.Tag(
        id=db_tag.id,
        name=db_tag.name,
        category=db_tag.category,
        usage_count=0
    )

@router.put("/{tag_id}", response_model=schemas.Tag)
def update_tag(
    tag_id: int, 
    tag_update: schemas.TagUpdate, 
    db: Session = Depends(get_db)
):
    """更新标签信息"""
    db_tag = crud.get_tag(db, tag_id)
    if not db_tag:
        raise HTTPException(status_code=404, detail="标签不存在")
    
    # 如果要更新名称，检查新名称是否已存在
    if tag_update.name and tag_update.name != db_tag.name:
        existing_tag = crud.get_tag_by_name(db, tag_update.name)
        if existing_tag:
            raise HTTPException(status_code=400, detail="标签名称已存在")
    
    updated_tag = crud.update_tag(db, tag_id, tag_update)
    
    # 获取使用次数
    usage_count = db.query(crud.models.material_tag).filter(
        crud.models.material_tag.c.tag_id == tag_id
    ).count()
    
    tag_dict = schemas.TagInDB.from_orm(updated_tag).dict()
    tag_dict["usage_count"] = usage_count
    
    return schemas.Tag(**tag_dict)

@router.delete("/{tag_id}")
def delete_tag(tag_id: int, db: Session = Depends(get_db)):
    """删除标签"""
    tag = crud.get_tag(db, tag_id)
    if not tag:
        raise HTTPException(status_code=404, detail="标签不存在")
    
    if not crud.delete_tag(db, tag_id):
        raise HTTPException(status_code=500, detail="删除标签失败")
    
    return {"success": True, "message": "标签已删除"}

@router.post("/merge", response_model=schemas.Tag)
def merge_tags(merge_data: schemas.TagMerge, db: Session = Depends(get_db)):
    """合并多个标签"""
    # 检查目标标签是否存在
    target_tag = crud.get_tag(db, merge_data.target_tag_id)
    if not target_tag:
        raise HTTPException(status_code=404, detail="目标标签不存在")
    
    # 检查源标签
    for tag_id in merge_data.source_tag_ids:
        if tag_id == merge_data.target_tag_id:
            continue  # 跳过目标标签
            
        source_tag = crud.get_tag(db, tag_id)
        if not source_tag:
            raise HTTPException(status_code=404, detail=f"源标签 (ID: {tag_id}) 不存在")
    
    try:
        # 开始事务
        db.begin_nested()
        
        # 将源标签的关联转移到目标标签
        for tag_id in merge_data.source_tag_ids:
            if tag_id == merge_data.target_tag_id:
                continue  # 跳过目标标签
                
            # 获取源标签的所有素材关联
            tag_materials = db.query(crud.models.material_tag).filter(
                crud.models.material_tag.c.tag_id == tag_id
            ).all()
            
            # 为每个素材添加目标标签关联
            for material_tag in tag_materials:
                # 检查目标标签关联是否已存在
                exists = db.query(crud.models.material_tag).filter(
                    crud.models.material_tag.c.material_id == material_tag.material_id,
                    crud.models.material_tag.c.tag_id == merge_data.target_tag_id
                ).first()
                
                if not exists:
                    # 添加新关联
                    db.execute(
                        crud.models.material_tag.insert().values(
                            material_id=material_tag.material_id,
                            tag_id=merge_data.target_tag_id,
                            confidence=material_tag.confidence
                        )
                    )
            
            # 删除源标签关联
            db.execute(
                crud.models.material_tag.delete().where(
                    crud.models.material_tag.c.tag_id == tag_id
                )
            )
            
            # 删除源标签
            db.delete(source_tag)
        
        # 提交事务
        db.commit()
        
        # 获取更新后的标签使用次数
        usage_count = db.query(crud.models.material_tag).filter(
            crud.models.material_tag.c.tag_id == merge_data.target_tag_id
        ).count()
        
        tag_dict = schemas.TagInDB.from_orm(target_tag).dict()
        tag_dict["usage_count"] = usage_count
        
        return schemas.Tag(**tag_dict)
        
    except Exception as e:
        db.rollback()
        logger.error(f"合并标签失败: {e}")
        raise HTTPException(status_code=500, detail=f"合并标签失败: {str(e)}")

@router.get("/{tag_id}/materials")
def get_tag_materials(
    tag_id: int, 
    skip: int = 0, 
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """获取使用此标签的所有素材"""
    tag = crud.get_tag(db, tag_id)
    if not tag:
        raise HTTPException(status_code=404, detail="标签不存在")
    
    # 查询使用此标签的素材
    materials = db.query(crud.models.Material).join(
        crud.models.material_tag,
        crud.models.Material.id == crud.models.material_tag.c.material_id
    ).filter(
        crud.models.material_tag.c.tag_id == tag_id
    ).order_by(
        crud.models.Material.source_video,
        crud.models.Material.timestamp
    ).offset(skip).limit(limit).all()
    
    return [
        {
            "id": material.id,
            "source_video": material.source_video,
            "frame_path": material.frame_path,
            "timestamp": material.timestamp
        } 
        for material in materials
    ]