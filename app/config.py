import os
import json
from typing import Dict, List, Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """应用配置，使用环境变量或.env文件"""
    APP_NAME: str = "视频素材管理系统"
    API_V1_STR: str = "/api/v1"
    
    # 文件路径
    VIDEO_FOLDER: str = Field("./app/static/uploads", env="VIDEO_FOLDER")
    OUTPUT_FOLDER: str = Field("./app/static/extracted_frames", env="OUTPUT_FOLDER")
    
    # 数据库设置
    SQLALCHEMY_DATABASE_URL: str = Field("sqlite:///./video_materials.db", env="DATABASE_URL")
    
    # 云API设置
    API_TYPE: str = Field("openai", env="API_TYPE")
    API_KEY: Optional[str] = Field(None, env="API_KEY")
    
    # 本地分类器设置
    USE_LOCAL_CLASSIFIER: bool = Field(False, env="USE_LOCAL_CLASSIFIER")
    LOCAL_MODEL_PATH: Optional[str] = Field(None, env="LOCAL_MODEL_PATH")
    LOCAL_MODEL_SIZE: str = Field("7b", env="LOCAL_MODEL_SIZE")
    LOCAL_TAGS_OUTPUT_FILE: str = Field("local_image_tags.json", env="LOCAL_TAGS_OUTPUT_FILE")
    GPU_LAYERS: int = Field(0, env="GPU_LAYERS")
    
    # Qwen-VL特定设置
    QWEN_VL_MMPROJ_PATH: Optional[str] = Field(None, env="QWEN_VL_MMPROJ_PATH")
    QWEN_VL_CLI_PATH: Optional[str] = Field(None, env="QWEN_VL_CLI_PATH")
    
    # 视频处理设置
    MAX_WORKERS: int = Field(4, env="MAX_WORKERS")
    MIN_SCENE_CHANGE_THRESHOLD: int = Field(30, env="MIN_SCENE_CHANGE_THRESHOLD")
    FRAME_SAMPLE_INTERVAL: int = Field(24, env="FRAME_SAMPLE_INTERVAL")
    QUALITY: int = Field(90, env="QUALITY")
    MAX_FRAMES_PER_VIDEO: int = Field(200, env="MAX_FRAMES_PER_VIDEO")
    
    # 默认标签
    CUSTOM_TAGS: List[str] = [
        "人物特写", "风景", "动作", "对话", "过渡", "特效",
        "室内", "室外", "白天", "夜晚", "城市", "自然",
        "快速", "慢动作", "情感", "叙事", "黑白", "彩色"
    ]
    
    # CORS设置
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

def load_config_from_json(config_file: str = "config.json") -> None:
    """从JSON文件加载配置并更新设置"""
    global settings
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                
                # 更新settings
                for key, value in config_data.items():
                    key_upper = key.upper()
                    if hasattr(settings, key_upper):
                        setattr(settings, key_upper, value)
                        
                # 处理嵌套结构
                if "extraction" in config_data:
                    extraction = config_data["extraction"]
                    if "min_scene_change_threshold" in extraction:
                        settings.MIN_SCENE_CHANGE_THRESHOLD = extraction["min_scene_change_threshold"]
                    if "frame_sample_interval" in extraction:
                        settings.FRAME_SAMPLE_INTERVAL = extraction["frame_sample_interval"]
                    if "quality" in extraction:
                        settings.QUALITY = extraction["quality"]
                    if "max_frames_per_video" in extraction:
                        settings.MAX_FRAMES_PER_VIDEO = extraction["max_frames_per_video"]
                        
                if "custom_tags" in config_data:
                    settings.CUSTOM_TAGS = config_data["custom_tags"]
                
                # 处理本地分类器设置
                if "local_classifier" in config_data:
                    local_config = config_data["local_classifier"]
                    if "use_local_classifier" in local_config:
                        settings.USE_LOCAL_CLASSIFIER = local_config["use_local_classifier"]
                    if "model_path" in local_config:
                        settings.LOCAL_MODEL_PATH = local_config["model_path"]
                    if "model_size" in local_config:
                        settings.LOCAL_MODEL_SIZE = local_config["model_size"]
                    if "tags_output_file" in local_config:
                        settings.LOCAL_TAGS_OUTPUT_FILE = local_config["tags_output_file"]
                    if "gpu_layers" in local_config:
                        settings.GPU_LAYERS = local_config["gpu_layers"]
                    # 新增Qwen-VL特定设置
                    if "mmproj_path" in local_config:
                        settings.QWEN_VL_MMPROJ_PATH = local_config["mmproj_path"]
                    if "cli_path" in local_config:
                        settings.QWEN_VL_CLI_PATH = local_config["cli_path"]
                    
        except Exception as e:
            print(f"加载配置文件失败: {e}")

# 创建必要的目录
os.makedirs(settings.VIDEO_FOLDER, exist_ok=True)
os.makedirs(settings.OUTPUT_FOLDER, exist_ok=True)

# 尝试加载JSON配置
load_config_from_json()