import os
import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from sqlalchemy.orm import Session

from app.config import settings
from app.core.frame_extractor import FrameExtractor
# 导入两个分类器
from app.core.image_classifier import ImageClassifier
from app.core.local_image_classifier import LocalImageClassifier
from app.db import crud, models

logger = logging.getLogger(__name__)

class MaterialProcessWorkflow:
    """视频素材处理工作流"""
    
    def __init__(self, db: Session):
        """
        初始化工作流
        
        参数:
            db: 数据库会话
        """
        self.db = db
        self.extractor = FrameExtractor(
            output_dir=settings.OUTPUT_FOLDER,
            min_scene_change_threshold=settings.MIN_SCENE_CHANGE_THRESHOLD,
            frame_sample_interval=settings.FRAME_SAMPLE_INTERVAL,
            quality=settings.QUALITY,
            max_frames=settings.MAX_FRAMES_PER_VIDEO
        )
        
        # 根据配置选择分类器
        if settings.USE_LOCAL_CLASSIFIER:
            logger.info("使用本地图像分类器")
            self.classifier = LocalImageClassifier(
                model_path=settings.LOCAL_MODEL_PATH,
                model_size=settings.LOCAL_MODEL_SIZE,
                max_workers=settings.MAX_WORKERS,
                custom_tags=settings.CUSTOM_TAGS,
                output_file=settings.LOCAL_TAGS_OUTPUT_FILE,
                gpu_layers=settings.GPU_LAYERS
            )
        else:
            logger.info("使用云端图像分类器")
            self.classifier = ImageClassifier(
                api_type=settings.API_TYPE,
                api_key=settings.API_KEY,
                max_workers=settings.MAX_WORKERS,
                custom_tags=settings.CUSTOM_TAGS
            )
    
    async def process_video(self, video_path: str, extract_only: bool = False) -> Dict[str, Any]:
        """
        处理单个视频
        
        参数:
            video_path: 视频文件路径
            extract_only: 仅提取帧，不进行分类
            
        返回:
            处理结果信息
        """
        start_time = time.time()
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            logger.error(f"视频不存在: {video_path}")
            return {"success": False, "error": "视频文件不存在", "video_path": video_path}
        
        video_name = Path(video_path).stem
        logger.info(f"开始处理视频: {video_name}")
        
        try:
            # 提取帧
            logger.info("步骤1: 提取关键帧")
            frames_info = self.extractor.extract_frames(video_path)
            
            if not frames_info:
                logger.warning(f"未能从视频中提取到任何帧: {video_path}")
                return {"success": False, "error": "未能提取帧", "video_path": video_path}
                
            # 获取所有提取的帧路径
            frame_paths = [frame_path for _, frame_path in frames_info]
            
            if extract_only:
                logger.info(f"仅提取模式: 已提取 {len(frame_paths)} 帧")
                return {
                    "success": True, 
                    "video_path": video_path,
                    "frames_count": len(frame_paths),
                    "frames": [{"timestamp": ts, "path": path} for ts, path in frames_info],
                    "extract_only": True
                }
            
            # 分类图像
            if isinstance(self.classifier, LocalImageClassifier):
                logger.info("步骤2: 本地模型分类图像")
            else:
                logger.info("步骤2: 云端分类图像")
                
            tag_results = self.classifier.classify_images(frame_paths)
            
            # 导入到数据库
            logger.info("步骤3: 保存到数据库")
            added_count = self._import_to_database(video_path, frames_info, tag_results)
            
            elapsed_time = time.time() - start_time
            logger.info(f"视频处理完成! 用时: {elapsed_time:.2f}秒, 添加了 {added_count} 个素材")
            
            return {
                "success": True,
                "video_path": video_path,
                "video_name": video_name,
                "frames_count": len(frame_paths),
                "imported_count": added_count,
                "processing_time": elapsed_time
            }
            
        except Exception as e:
            logger.error(f"处理视频出错: {e}", exc_info=True)
            return {"success": False, "error": str(e), "video_path": video_path}
    
    def _import_to_database(
        self, 
        video_path: str, 
        frames_info: List[Tuple[float, str]], 
        tag_results: Dict[str, Any]
    ) -> int:
        """
        导入处理结果到数据库
        
        参数:
            video_path: 视频文件路径
            frames_info: (时间码, 文件路径)列表
            tag_results: 标签结果
            
        返回:
            添加的素材数量
        """
        added_count = 0
        
        for timestamp, frame_path in frames_info:
            # 检查素材是否已存在
            existing_material = crud.get_material_by_path(self.db, frame_path)
            if existing_material:
                logger.debug(f"素材已存在: {frame_path}")
                continue
                
            # 创建素材记录
            try:
                material = crud.create_material(self.db, {
                    "source_video": video_path,
                    "frame_path": frame_path,
                    "timestamp": timestamp,
                    "description": ""
                })
                
                # 如果有标签结果，添加标签
                if frame_path in tag_results:
                    tags_info = tag_results[frame_path]
                    tag_ids = []
                    
                    # 处理标准格式的标签
                    all_tags = []
                    
                    # 从tags字段获取标签
                    if isinstance(tags_info, dict) and "tags" in tags_info:
                        if isinstance(tags_info["tags"], list):
                            all_tags.extend(tags_info["tags"])
                    
                    # 从scene字段获取标签
                    if isinstance(tags_info, dict) and "scene" in tags_info:
                        if isinstance(tags_info["scene"], str):
                            all_tags.append(tags_info["scene"])
                        elif isinstance(tags_info["scene"], list):
                            all_tags.extend(tags_info["scene"])
                    
                    # 从elements字段获取标签
                    if isinstance(tags_info, dict) and "elements" in tags_info:
                        if isinstance(tags_info["elements"], list):
                            all_tags.extend(tags_info["elements"])
                    
                    # 从lighting字段获取标签
                    if isinstance(tags_info, dict) and "lighting" in tags_info:
                        if isinstance(tags_info["lighting"], str):
                            all_tags.append(tags_info["lighting"])
                        elif isinstance(tags_info["lighting"], list):
                            all_tags.extend(tags_info["lighting"])
                    
                    # 从mood字段获取标签
                    if isinstance(tags_info, dict) and "mood" in tags_info:
                        if isinstance(tags_info["mood"], str):
                            all_tags.append(tags_info["mood"])
                        elif isinstance(tags_info["mood"], list):
                            all_tags.extend(tags_info["mood"])
                    
                    # 处理直接是列表的情况
                    if isinstance(tags_info, list):
                        all_tags.extend(tags_info)
                    
                    # 清理标签列表
                    clean_tags = []
                    for tag in all_tags:
                        if isinstance(tag, str) and tag.strip():
                            clean_tags.append(tag.strip())
                    
                    # 去重
                    clean_tags = list(set(clean_tags))
                    
                    # 为素材添加标签
                    for tag_name in clean_tags:
                        # 获取或创建标签
                        tag = crud.get_tag_by_name(self.db, tag_name)
                        if not tag:
                            # 猜测标签类别
                            tag_category = self.classifier.guess_tag_category(tag_name)
                            tag = crud.create_tag(self.db, {"name": tag_name, "category": tag_category})
                        
                        # 获取置信度
                        confidence = 1.0
                        if isinstance(tags_info, dict) and "confidence" in tags_info:
                            if isinstance(tags_info["confidence"], dict) and tag_name in tags_info["confidence"]:
                                confidence = float(tags_info["confidence"][tag_name])
                        
                        # 添加标签关联
                        crud.add_material_tags(self.db, material.id, [tag.id], confidence)
                
                added_count += 1
                
                # 每50个素材记录日志
                if added_count % 50 == 0:
                    logger.info(f"已导入 {added_count} 个素材...")
                
            except Exception as e:
                logger.error(f"导入素材出错 {frame_path}: {e}")
                self.db.rollback()
        
        logger.info(f"导入完成: 成功添加 {added_count} 个素材")
        return added_count