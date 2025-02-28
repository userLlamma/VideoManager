from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import logging
import shutil
from pathlib import Path
import asyncio
from datetime import datetime

from app.db.database import get_db
from app.core.workflow import MaterialProcessWorkflow
from app.api.deps import get_workflow
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    extract_only: bool = Form(False),
    workflow: MaterialProcessWorkflow = Depends(get_workflow)
):
    """上传视频并处理"""
    # 检查文件类型
    allowed_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的文件类型。允许的类型: {', '.join(allowed_extensions)}"
        )
    
    # 生成保存路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    save_path = os.path.join(settings.VIDEO_FOLDER, filename)
    
    # 保存上传的文件
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"保存上传文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存文件失败: {str(e)}")
    finally:
        await file.close()
    
    # 在后台处理视频
    async def process_video_task():
        try:
            await workflow.process_video(save_path, extract_only)
        except Exception as e:
            logger.error(f"处理视频失败: {e}")
    
    background_tasks.add_task(process_video_task)
    
    return {
        "success": True,
        "message": "视频已上传，正在处理中",
        "video_path": save_path,
        "extract_only": extract_only
    }

@router.post("/process")
async def process_existing_video(
    video_path: str,
    extract_only: bool = False,
    workflow: MaterialProcessWorkflow = Depends(get_workflow)
):
    """处理已存在的视频文件"""
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="视频文件不存在")
    
    # 处理视频
    result = await workflow.process_video(video_path, extract_only)
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result

@router.post("/batch")
async def batch_process_videos(
    video_folder: str = Form(...),
    extract_only: bool = Form(False),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    workflow: MaterialProcessWorkflow = Depends(get_workflow)
):
    """批量处理文件夹中的视频"""
    if not os.path.exists(video_folder) or not os.path.isdir(video_folder):
        raise HTTPException(status_code=404, detail="指定的文件夹不存在")
    
    # 查找所有视频文件
    video_files = []
    allowed_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    
    for root, _, files in os.walk(video_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in allowed_extensions):
                video_files.append(os.path.join(root, file))
    
    if not video_files:
        raise HTTPException(status_code=404, detail="文件夹中未找到视频文件")
    
    # 在后台处理所有视频
    async def process_videos_task():
        results = []
        for video_path in video_files:
            try:
                result = await workflow.process_video(video_path, extract_only)
                results.append(result)
            except Exception as e:
                logger.error(f"处理视频 {video_path} 失败: {e}")
                results.append({
                    "success": False,
                    "video_path": video_path,
                    "error": str(e)
                })
        
        logger.info(f"批量处理完成, 成功: {sum(1 for r in results if r['success'])}/{len(results)}")
    
    background_tasks.add_task(process_videos_task)
    
    return {
        "success": True,
        "message": f"开始处理 {len(video_files)} 个视频文件",
        "video_count": len(video_files),
        "video_folder": video_folder,
        "extract_only": extract_only
    }