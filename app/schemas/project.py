from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectCreate(ProjectBase):
    pass

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class ProjectInDB(ProjectBase):
    id: int
    created_date: datetime
    
    class Config:
        from_attributes = True

class ProjectMaterialAdd(BaseModel):
    material_ids: List[int]
    notes: Optional[str] = None

class Project(ProjectInDB):
    material_count: int = 0
    
    class Config:
        from_attributes = True
