from typing import List, Optional
from pydantic import BaseModel

class TagBase(BaseModel):
    name: str
    category: Optional[str] = None

class TagCreate(TagBase):
    pass

class TagUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None

class TagInDB(TagBase):
    id: int
    
    class Config:
        from_attributes = True

class Tag(TagInDB):
    usage_count: int = 0
    
    class Config:
        from_attributes = True

class TagMerge(BaseModel):
    source_tag_ids: List[int]
    target_tag_id: int