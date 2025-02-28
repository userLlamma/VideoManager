from sqlalchemy import Column, Integer, String, Float, ForeignKey, Table, DateTime, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

from .database import Base

# 素材-标签关联表
material_tag = Table(
    "material_tag",
    Base.metadata,
    Column("material_id", Integer, ForeignKey("materials.id"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id"), primary_key=True),
    Column("confidence", Float, default=1.0)
)

# 项目-素材关联表
project_material = Table(
    "project_material",
    Base.metadata,
    Column("project_id", Integer, ForeignKey("projects.id"), primary_key=True),
    Column("material_id", Integer, ForeignKey("materials.id"), primary_key=True),
    Column("added_date", DateTime, default=func.now()),
    Column("notes", Text, nullable=True)
)

class Material(Base):
    """素材模型 - 对应视频中提取的帧"""
    __tablename__ = "materials"
    
    id = Column(Integer, primary_key=True, index=True)
    source_video = Column(String, nullable=False)
    frame_path = Column(String, nullable=False, unique=True)
    timestamp = Column(Float, nullable=False)
    description = Column(Text, nullable=True)
    added_date = Column(DateTime, default=func.now())
    
    # 关系
    tags = relationship("Tag", secondary=material_tag, back_populates="materials")
    projects = relationship("Project", secondary=project_material, back_populates="materials")

class Tag(Base):
    """标签模型 - 素材的分类标签"""
    __tablename__ = "tags"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    category = Column(String, nullable=True)
    
    # 关系
    materials = relationship("Material", secondary=material_tag, back_populates="tags")
    
    class Config:
        orm_mode = True

class Project(Base):
    """项目模型 - 素材集合"""
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=True)
    created_date = Column(DateTime, default=func.now())
    
    # 关系
    materials = relationship("Material", secondary=project_material, back_populates="projects")