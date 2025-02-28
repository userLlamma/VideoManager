from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class TagInfo(BaseModel):
    id: int
    name: str
    category: Optional[str] = None
    confidence: float = 1.0

class MaterialBase(BaseModel):
    source_video: str
    frame_path: str
    timestamp: float
    description: Optional[str] = None

class MaterialCreate(MaterialBase):
    pass

class MaterialUpdate(BaseModel):
    source_video: Optional[str] = None
    description: Optional[str] = None

class MaterialInDB(MaterialBase):
    id: int
    added_date: datetime
    
    class Config:
        orm_mode = True

class Material(MaterialInDB):
    tags: List[TagInfo] = []
    
    class Config:
        orm_mode = True

class MaterialWithTags(Material):
    tags: List[TagInfo] = []
    projects: List[int] = []
    
    class Config:
        orm_mode = True