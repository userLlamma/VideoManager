from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from . import models
from app.schemas import material as material_schema
from app.schemas import tag as tag_schema
from app.schemas import project as project_schema

# 素材相关CRUD操作
def get_materials(
    db: Session, 
    skip: int = 0, 
    limit: int = 100,
    search: Optional[str] = None,
    tag_ids: Optional[List[int]] = None
) -> List[models.Material]:
    """获取素材列表，支持搜索和标签过滤"""
    query = db.query(models.Material)
    
    # 应用搜索过滤
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                models.Material.source_video.like(search_term),
                models.Material.description.like(search_term),
                models.Material.id.in_(
                    db.query(models.material_tag.c.material_id)
                    .join(models.Tag, models.material_tag.c.tag_id == models.Tag.id)
                    .filter(models.Tag.name.like(search_term))
                )
            )
        )
    
    # 应用标签过滤
    if tag_ids and len(tag_ids) > 0:
        # 查找包含所有指定标签的素材
        for tag_id in tag_ids:
            query = query.filter(
                models.Material.id.in_(
                    db.query(models.material_tag.c.material_id)
                    .filter(models.material_tag.c.tag_id == tag_id)
                )
            )
    
    # 应用分页并返回结果
    return query.order_by(models.Material.source_video, models.Material.timestamp).offset(skip).limit(limit).all()

def get_material(db: Session, material_id: int) -> Optional[models.Material]:
    """通过ID获取单个素材"""
    return db.query(models.Material).filter(models.Material.id == material_id).first()

def get_material_by_path(db: Session, frame_path: str) -> Optional[models.Material]:
    """通过路径获取素材"""
    return db.query(models.Material).filter(models.Material.frame_path == frame_path).first()

def create_material(db: Session, material: material_schema.MaterialCreate) -> models.Material:
    """创建新素材"""
    db_material = models.Material(
        source_video=material.source_video,
        frame_path=material.frame_path,
        timestamp=material.timestamp,
        description=material.description,
        added_date=datetime.now()
    )
    db.add(db_material)
    db.commit()
    db.refresh(db_material)
    return db_material

def update_material(
    db: Session, 
    material_id: int, 
    material_update: material_schema.MaterialUpdate
) -> Optional[models.Material]:
    """更新素材信息"""
    db_material = get_material(db, material_id)
    if db_material:
        update_data = material_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_material, key, value)
        db.commit()
        db.refresh(db_material)
    return db_material

def delete_material(db: Session, material_id: int) -> bool:
    """删除素材"""
    db_material = get_material(db, material_id)
    if db_material:
        # 删除关联关系
        db.execute(models.material_tag.delete().where(models.material_tag.c.material_id == material_id))
        db.execute(models.project_material.delete().where(models.project_material.c.material_id == material_id))
        
        # 删除素材
        db.delete(db_material)
        db.commit()
        return True
    return False

# 标签相关CRUD操作
def get_tags(
    db: Session, 
    skip: int = 0, 
    limit: int = 100,
    category: Optional[str] = None
) -> List[models.Tag]:
    """获取标签列表，可按类别过滤"""
    query = db.query(models.Tag)
    
    if category:
        query = query.filter(models.Tag.category == category)
    
    return query.order_by(models.Tag.category, models.Tag.name).offset(skip).limit(limit).all()

def get_tag(db: Session, tag_id: int) -> Optional[models.Tag]:
    """通过ID获取标签"""
    return db.query(models.Tag).filter(models.Tag.id == tag_id).first()

def get_tag_by_name(db: Session, name: str) -> Optional[models.Tag]:
    """通过名称获取标签"""
    return db.query(models.Tag).filter(func.lower(models.Tag.name) == func.lower(name)).first()

def create_tag(db: Session, tag: tag_schema.TagCreate) -> models.Tag:
    """创建新标签"""
    db_tag = models.Tag(name=tag.name, category=tag.category)
    db.add(db_tag)
    db.commit()
    db.refresh(db_tag)
    return db_tag

def update_tag(db: Session, tag_id: int, tag_update: tag_schema.TagUpdate) -> Optional[models.Tag]:
    """更新标签信息"""
    db_tag = get_tag(db, tag_id)
    if db_tag:
        update_data = tag_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_tag, key, value)
        db.commit()
        db.refresh(db_tag)
    return db_tag

def delete_tag(db: Session, tag_id: int) -> bool:
    """删除标签"""
    db_tag = get_tag(db, tag_id)
    if db_tag:
        # 删除关联关系
        db.execute(models.material_tag.delete().where(models.material_tag.c.tag_id == tag_id))
        
        # 删除标签
        db.delete(db_tag)
        db.commit()
        return True
    return False

def get_tag_categories(db: Session) -> List[str]:
    """获取所有标签类别"""
    result = db.query(models.Tag.category).distinct().all()
    return [r[0] for r in result if r[0]]  # 过滤掉None

# 项目相关CRUD操作
def get_projects(db: Session, skip: int = 0, limit: int = 100) -> List[models.Project]:
    """获取项目列表"""
    return db.query(models.Project).order_by(models.Project.name).offset(skip).limit(limit).all()

def get_project(db: Session, project_id: int) -> Optional[models.Project]:
    """通过ID获取项目"""
    return db.query(models.Project).filter(models.Project.id == project_id).first()

def get_project_by_name(db: Session, name: str) -> Optional[models.Project]:
    """通过名称获取项目"""
    return db.query(models.Project).filter(func.lower(models.Project.name) == func.lower(name)).first()

def create_project(db: Session, project: project_schema.ProjectCreate) -> models.Project:
    """创建新项目"""
    db_project = models.Project(
        name=project.name,
        description=project.description,
        created_date=datetime.now()
    )
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

def update_project(
    db: Session, 
    project_id: int, 
    project_update: project_schema.ProjectUpdate
) -> Optional[models.Project]:
    """更新项目信息"""
    db_project = get_project(db, project_id)
    if db_project:
        update_data = project_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_project, key, value)
        db.commit()
        db.refresh(db_project)
    return db_project

def delete_project(db: Session, project_id: int) -> bool:
    """删除项目"""
    db_project = get_project(db, project_id)
    if db_project:
        # 删除关联关系
        db.execute(models.project_material.delete().where(models.project_material.c.project_id == project_id))
        
        # 删除项目
        db.delete(db_project)
        db.commit()
        return True
    return False

# 素材标签关系操作
def add_material_tags(
    db: Session, 
    material_id: int, 
    tag_ids: List[int],
    confidence: float = 1.0
) -> bool:
    """为素材添加标签"""
    try:
        # 检查素材是否存在
        material = get_material(db, material_id)
        if not material:
            return False
            
        # 添加标签关联
        for tag_id in tag_ids:
            # 检查标签是否存在
            tag = get_tag(db, tag_id)
            if tag:
                # 添加关联
                db.execute(
                    models.material_tag.insert().values(
                        material_id=material_id,
                        tag_id=tag_id,
                        confidence=confidence
                    )
                )
        
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"添加素材标签失败: {e}")
        return False

def remove_material_tag(db: Session, material_id: int, tag_id: int) -> bool:
    """移除素材的标签"""
    try:
        result = db.execute(
            models.material_tag.delete().where(
                and_(
                    models.material_tag.c.material_id == material_id,
                    models.material_tag.c.tag_id == tag_id
                )
            )
        )
        db.commit()
        return result.rowcount > 0
    except Exception as e:
        db.rollback()
        print(f"移除素材标签失败: {e}")
        return False

def get_material_tags(db: Session, material_id: int) -> List[Dict[str, Any]]:
    """获取素材的所有标签及置信度"""
    result = db.query(
        models.Tag.id, 
        models.Tag.name, 
        models.Tag.category,
        models.material_tag.c.confidence
    ).join(
        models.material_tag, 
        models.Tag.id == models.material_tag.c.tag_id
    ).filter(
        models.material_tag.c.material_id == material_id
    ).all()
    
    return [
        {
            "id": r[0],
            "name": r[1],
            "category": r[2],
            "confidence": r[3]
        } 
        for r in result
    ]

# 项目素材关系操作
def add_material_to_project(
    db: Session, 
    project_id: int, 
    material_id: int,
    notes: Optional[str] = None
) -> bool:
    """将素材添加到项目"""
    try:
        # 检查项目和素材是否存在
        project = get_project(db, project_id)
        material = get_material(db, material_id)
        
        if not project or not material:
            return False
            
        # 添加关联
        db.execute(
            models.project_material.insert().values(
                project_id=project_id,
                material_id=material_id,
                added_date=datetime.now(),
                notes=notes
            )
        )
        
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"添加素材到项目失败: {e}")
        return False

def remove_material_from_project(db: Session, project_id: int, material_id: int) -> bool:
    """从项目中移除素材"""
    try:
        result = db.execute(
            models.project_material.delete().where(
                and_(
                    models.project_material.c.project_id == project_id,
                    models.project_material.c.material_id == material_id
                )
            )
        )
        db.commit()
        return result.rowcount > 0
    except Exception as e:
        db.rollback()
        print(f"从项目移除素材失败: {e}")
        return False

def get_project_materials(db: Session, project_id: int, skip: int = 0, limit: int = 100) -> List[models.Material]:
    """获取项目中的所有素材"""
    return db.query(models.Material).join(
        models.project_material,
        models.Material.id == models.project_material.c.material_id
    ).filter(
        models.project_material.c.project_id == project_id
    ).order_by(
        models.Material.source_video, 
        models.Material.timestamp
    ).offset(skip).limit(limit).all()