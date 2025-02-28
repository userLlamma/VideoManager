from fastapi import APIRouter

from app.api.endpoints import materials, tags, projects, processing

api_router = APIRouter()
api_router.include_router(materials.router, prefix="/materials", tags=["materials"])
api_router.include_router(tags.router, prefix="/tags", tags=["tags"])
api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(processing.router, prefix="/processing", tags=["processing"])