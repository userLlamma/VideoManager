import os
import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from sqlalchemy.orm import Session

from app.config import settings
from app.core.frame_extractor import FrameExtractor
# Import the classifiers
from app.core.image_classifier import ImageClassifier
from app.core.local_image_classifier  import QwenVLClassifier  # Use our new implementation
from app.db import crud, models
from app.schemas.material import MaterialCreate

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
            logger.info("使用本地图像分类器 (Qwen-VL)")
            self.classifier = QwenVLClassifier(
                model_path=settings.LOCAL_MODEL_PATH,
                mmproj_path=settings.QWEN_VL_MMPROJ_PATH,
                cli_path=settings.QWEN_VL_CLI_PATH,
                model_size=settings.LOCAL_MODEL_SIZE,
                max_workers=min(settings.MAX_WORKERS, 3),  # Limit workers for local classification
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
            if isinstance(self.classifier, QwenVLClassifier):
                logger.info("步骤2: 使用本地Qwen-VL模型分类图像")
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
    
    def _process_tags_from_result(self, tags_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理从分类器得到的标签结果，提取有用的标签
        
        参数:
            tags_info: 分类器返回的标签信息
                
        返回:
            处理后的标签列表，每个标签包含名称和类别
        """
        import re
        
        processed_tags = []
        confidence_dict = {}
        
        # 定义字段到类别的映射
        field_to_category = {
            "tags": None,  # 将根据标签内容推断
            "scene": "场景",
            "elements": "元素",
            "lighting": "光照",
            "mood": "情绪",
            "composition": "构图"
        }
        
        # 从各个字段提取标签
        for field, category in field_to_category.items():
            if isinstance(tags_info, dict) and field in tags_info:
                # 处理数组类型的字段
                if isinstance(tags_info[field], list):
                    for item in tags_info[field]:
                        if isinstance(item, str) and item.strip() and item.strip() not in ["解析错误", "未识别", "处理错误"]:
                            # 清理标签文本
                            clean_tag = re.sub(r'[\"\'"""''\[\]\{\}]', '', item.strip())
                            if clean_tag:
                                # 如果类别为None，使用标签猜测函数
                                tag_category = category
                                if category is None and hasattr(self.classifier, 'guess_tag_category'):
                                    tag_category = self.classifier.guess_tag_category(clean_tag)
                                
                                processed_tags.append({
                                    "name": clean_tag, 
                                    "category": tag_category
                                })
                # 处理字符串类型的字段
                elif isinstance(tags_info[field], str) and tags_info[field].strip():
                    clean_tag = re.sub(r'[\"\'"""''\[\]\{\}]', '', tags_info[field].strip())
                    if clean_tag:
                        processed_tags.append({
                            "name": clean_tag,
                            "category": category
                        })
        
        # 提取置信度信息
        if isinstance(tags_info, dict) and "confidence" in tags_info:
            if isinstance(tags_info["confidence"], dict):
                confidence_dict = tags_info["confidence"]
        
        # 对标签进行去重和质量过滤
        unique_tags = []
        seen_names = set()
        min_confidence = getattr(self, 'min_confidence_threshold', 0.5)
        
        for tag in processed_tags:
            # 跳过过于简单或无意义的标签
            if len(tag["name"]) < 2 or tag["name"].lower() in ["的", "了", "和", "与", "在", "是"]:
                continue
                
            # 检查标签是否已存在（不区分大小写）
            if tag["name"].lower() not in seen_names:
                seen_names.add(tag["name"].lower())
                
                # 添加置信度
                if tag["name"] in confidence_dict:
                    tag["confidence"] = confidence_dict[tag["name"]]
                    # 过滤低置信度标签
                    if tag["confidence"] < min_confidence:
                        continue
                else:
                    tag["confidence"] = 1.0
                    
                unique_tags.append(tag)
        
        # 如果没有提取到任何标签，添加一个默认标签
        if not unique_tags:
            unique_tags.append({
                "name": "未分类", 
                "category": "其他",
                "confidence": 1.0
            })
        
        # 按类别排序标签
        category_order = ["场景", "元素", "光照", "情绪", "构图", "其他"]
        
        def get_category_order(tag):
            category = tag.get("category", "其他")
            try:
                return category_order.index(category)
            except ValueError:
                return len(category_order)
        
        # 按类别和置信度排序
        return sorted(unique_tags, key=lambda t: (get_category_order(t), -t.get("confidence", 0)))
    
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
        from app.schemas.material import MaterialCreate
        from app.schemas.tag import TagCreate
        
        added_count = 0
        error_count = 0
        
        for timestamp, frame_path in frames_info:
            # 检查素材是否已存在
            existing_material = crud.get_material_by_path(self.db, frame_path)
            if existing_material:
                logger.debug(f"素材已存在: {frame_path}")
                continue
                
            # 创建素材记录
            try:
                # 正确创建MaterialCreate对象
                material_data = MaterialCreate(
                    source_video=video_path,
                    frame_path=frame_path,
                    timestamp=timestamp,
                    description=""
                )
                material = crud.create_material(self.db, material_data)
                
                # 如果有标签结果，添加标签
                if frame_path in tag_results:
                    tags_info = tag_results[frame_path]
                    
                    # 使用增强的标签处理方法
                    processed_tags = self._process_tags_from_result(tags_info)
                    
                    # 为素材添加标签
                    for tag_data in processed_tags:
                        tag_name = tag_data["name"]
                        tag_category = tag_data.get("category", "其他")
                        tag_confidence = tag_data.get("confidence", 1.0)
                        
                        # 获取或创建标签
                        tag = crud.get_tag_by_name(self.db, tag_name)
                        if not tag:
                            # 创建新标签
                            tag_create_data = TagCreate(
                                name=tag_name,
                                category=tag_category
                            )
                            tag = crud.create_tag(self.db, tag_create_data)
                        
                        # 添加标签关联
                        if tag and tag.id:
                            crud.add_material_tags(self.db, material.id, [tag.id], tag_confidence)
                
                added_count += 1
                
                # 每50个素材记录日志
                if added_count % 50 == 0:
                    logger.info(f"已导入 {added_count} 个素材...")
                
            except Exception as e:
                error_count += 1
                logger.error(f"导入素材出错 {frame_path}: {e}")
                self.db.rollback()
        
        logger.info(f"导入完成: 成功添加 {added_count} 个素材, 失败 {error_count} 个")
        return added_count