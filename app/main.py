from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import os

from app.api import api_router
from app.config import settings
from app.db.database import engine, Base

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# 创建数据库表
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="视频素材管理系统",
    description="用于视频关键帧提取、分类和检索的API",
    version="1.0.0",
)

# 配置CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# 确保静态目录存在
os.makedirs(settings.OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.join("app", "static"), exist_ok=True)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 如果OUTPUT_FOLDER不在app/static下，则额外挂载
if not settings.OUTPUT_FOLDER.startswith("app/static"):
    extract_dir_name = os.path.basename(settings.OUTPUT_FOLDER)
    app.mount("/extracted_frames", StaticFiles(directory=settings.OUTPUT_FOLDER), name="extracted_frames")
    logger.info(f"挂载额外静态目录: {settings.OUTPUT_FOLDER} -> /extracted_frames")

# 注册API路由
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def read_root():
    return {"message": "视频素材管理系统API", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}