import os
import json
import time
import base64
import logging
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class LocalImageClassifier:
    """本地图像分类系统 - 使用llama.cpp加载Qwen2.5-VL模型"""
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        model_size: str = "7b",
        max_workers: int = 5,
        custom_tags: Optional[List[str]] = None,
        output_file: Optional[str] = None,
        gpu_layers: int = 0  # 0表示纯CPU推理，大于0时使用GPU加速
    ):
        """
        初始化本地图像分类器
        
        参数:
            model_path: Qwen-VL模型路径，若为None则使用默认路径
            model_size: 模型尺寸 ("1.5b", "7b", "72b")
            max_workers: 并行处理的最大线程数
            custom_tags: 自定义标签列表，用于引导模型生成特定领域的标签
            output_file: 输出的JSON文件路径（可选）
            gpu_layers: GPU加速的层数，0表示仅CPU推理
        """
        self.model_size = model_size
        self.model_path = model_path or self._get_default_model_path()
        self.max_workers = max_workers
        self.output_file = output_file
        self.custom_tags = custom_tags or []
        self.gpu_layers = gpu_layers
        
        # 初始化标签存储
        self.tags_data = {}
        if output_file and os.path.exists(output_file):
            self._load_existing_tags()
            
        # 初始化llama.cpp模型
        self._init_llama_model()
    
    def _get_default_model_path(self) -> str:
        """获取默认模型路径"""
        home = str(Path.home())
        # 使用正确的模型文件名
        model_filename = f"Qwen2-VL-7B-Instruct-Q4_K_M.gguf"
        return os.path.join(home, "models", "qwen-vl", model_filename)
        
    def _init_llama_model(self) -> None:
        """初始化llama.cpp模型"""
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.error("请安装llama-cpp-python: pip install llama-cpp-python")
            raise ImportError("缺少必要的库: llama-cpp-python")
            
        if not os.path.exists(self.model_path):
            logger.error(f"模型文件不存在: {self.model_path}")
            logger.info(f"请下载Qwen2-VL模型并放置于: {self.model_path}")
            logger.info(f"下载地址参考: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-GGUF")
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
        logger.info(f"正在加载模型: {self.model_path}")
        
        # 根据模型大小设置合适的参数，增大上下文窗口
        if self.model_size == "1.5b":
            ctx_size = 8192
            batch_size = 512
        elif self.model_size == "7b":
            ctx_size = 16384  # 增大上下文窗口
            batch_size = 256
        else:  # 72b
            ctx_size = 16384
            batch_size = 128
            
        try:
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=ctx_size,
                n_batch=batch_size,
                n_gpu_layers=self.gpu_layers,
                verbose=False
            )
            logger.info(f"模型加载成功: Qwen2.5-VL-{self.model_size}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _load_existing_tags(self) -> None:
        """加载已存在的标签数据"""
        if not self.output_file or not os.path.exists(self.output_file):
            return
            
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                self.tags_data = json.load(f)
            logger.info(f"已加载{len(self.tags_data)}个已标记的图像")
        except json.JSONDecodeError:
            logger.warning("标签文件格式错误，将创建新文件")
            self.tags_data = {}
    
    def _save_tags(self) -> None:
        """保存标签数据到文件"""
        if not self.output_file:
            return
            
        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.tags_data, f, ensure_ascii=False, indent=2)
    
    def classify_images(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        批量分类图像
        
        参数:
            image_paths: 图像路径列表
            
        返回:
            图像路径到标签的映射
        """
        # 过滤掉已处理的图像
        new_images = [img for img in image_paths if img not in self.tags_data]
        if not new_images:
            logger.info("所有图像已标记")
            return self.tags_data
        
        logger.info(f"开始处理{len(new_images)}个新图像...")
        
        try:
            # 使用线程池并行处理
            results_count = 0
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._classify_single_image, img): img for img in new_images}
                
                for future in futures:
                    img_path = futures[future]
                    try:
                        tags = future.result()
                        if tags:
                            self.tags_data[img_path] = tags
                            results_count += 1
                            
                            # 每10张图像保存一次结果
                            if results_count % 10 == 0:
                                logger.info(f"已处理 {results_count}/{len(new_images)} 张图像")
                                self._save_tags()
                    except Exception as e:
                        logger.error(f"处理图像出错 {img_path}: {e}")
                        # 添加带有错误信息的回退结果
                        video_name = os.path.basename(os.path.dirname(img_path))
                        self.tags_data[img_path] = {
                            "tags": ["处理错误"],
                            "source_video": video_name,
                            "error": str(e)
                        }
            
        except Exception as e:
            logger.error(f"批量处理失败，使用回退方案: {e}")
            # 使用简单的基于文件名的回退方案
            for img_path in new_images:
                if img_path not in self.tags_data:
                    video_name = os.path.basename(os.path.dirname(img_path))
                    # 从文件名提取简单信息作为标签
                    filename = os.path.basename(img_path)
                    tags = [part for part in filename.split('_') if len(part) > 2]
                    
                    self.tags_data[img_path] = {
                        "tags": tags or ["自动标记"],
                        "source_video": video_name,
                        "scene": "未知场景",
                        "elements": ["自动生成"],
                        "method": "回退方案"
                    }
        
        # 最终保存结果
        self._save_tags()
        logger.info(f"完成分类，共处理 {len(new_images)} 张图像")
        return self.tags_data
    
    def _convert_image_to_base64(self, image_path: str) -> str:
        """将图像转换为base64格式"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _prepare_image_for_model(self, image_path: str, max_size: int = 224, quality: int = 30) -> str:
        """
        准备图像格式给模型使用 - 使用极小尺寸和高压缩率
        
        参数:
            image_path: 图像文件路径
            max_size: 图像最大边长
            quality: JPEG压缩质量(1-100)
            
        返回:
            处理后的图像数据
        """
        try:
            from PIL import Image
            import io
            import base64
            
            # 读取原始图像
            with Image.open(image_path) as img:
                # 调整图像大小 
                ratio = min(max_size / img.width, max_size / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
                
                # 转换为RGB并压缩
                buffer = io.BytesIO()
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(buffer, format="JPEG", quality=quality)
                buffer.seek(0)
                
                # 转换为base64
                base64_image = base64.b64encode(buffer.read()).decode('utf-8')
                return f"![image](data:image/jpeg;base64,{base64_image})"
        except Exception as e:
            logger.error(f"图像准备失败: {e}")
            raise
    
    def _classify_single_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        对单个图像进行分类 - 使用最简短的提示和极度压缩的图像
        
        参数:
            image_path: 图像文件路径
            
        返回:
            图像标签数据，包含source_video属性
        """
        if not os.path.exists(image_path):
            logger.warning(f"图像不存在: {image_path}")
            return None
        
        try:
            # 使用极小尺寸和高压缩率
            image_data = self._prepare_image_for_model(image_path, max_size=224, quality=30)
            
            # 最简化的提示
            prompt = f"""<|im_start|>user
分析图像并提供: 1)3-5个标签 2)场景类型 3)主要内容。JSON格式返回。

{image_data}
<|im_end|>

<|im_start|>assistant
"""

            try:
                # 使用模型推理
                response = self.model.create_completion(
                    prompt=prompt,
                    max_tokens=512,  # 减少输出token数
                    temperature=0.1,
                    top_p=0.9,
                    stop=["<|im_end|>"]
                )
                
                # 获取生成的文本
                generated_text = response["choices"][0]["text"]
                
                # 解析JSON
                result = self._extract_json_from_text(generated_text)
                
                # 提取视频源信息并添加到结果中
                video_name = os.path.basename(os.path.dirname(image_path))
                if not video_name or video_name == '.':
                    # 如果无法从路径获取，使用图像文件名的一部分
                    parts = os.path.basename(image_path).split('_')
                    video_name = '_'.join(parts[:-1]) if len(parts) > 1 else "unknown_video"
                    
                # 确保结果是字典类型并添加source_video属性
                if not isinstance(result, dict):
                    result = {"tags": ["无法解析"], "error": "模型输出格式不正确"}
                
                result["source_video"] = video_name
                return result
                
            except Exception as e:
                logger.error(f"模型推理失败: {e}")
                # 创建包含source_video的错误结果
                video_name = os.path.basename(os.path.dirname(image_path))
                return {
                    "tags": ["处理错误"],
                    "source_video": video_name,
                    "error": str(e)
                }
                
        except Exception as e:
            logger.error(f"分类失败: {e}")
            # 即使在完全失败时也返回带有source_video的结果
            return {
                "tags": ["处理错误"],
                "source_video": "unknown_video",
                "error": str(e)
            }
            
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """从生成的文本中提取JSON信息"""
        import re
        
        # 尝试找到JSON块
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        json_match = re.search(json_pattern, text)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # 尝试直接解析整个文本
            json_str = text
            
        # 清理文本并尝试解析
        try:
            # 移除不是JSON的前缀和后缀
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                clean_json = json_str[start_idx:end_idx]
                return json.loads(clean_json)
            else:
                # 如果没有找到JSON结构，创建一个简单的标签集
                return {"tags": ["解析错误"], "raw_text": text}
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            
            # 尝试使用正则表达式提取键值对
            tags = re.findall(r'"tags"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            scene = re.findall(r'"scene"\s*:\s*"(.*?)"', text)
            elements = re.findall(r'"elements"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            lighting = re.findall(r'"lighting"\s*:\s*"(.*?)"', text)
            mood = re.findall(r'"mood"\s*:\s*"(.*?)"', text)
            
            result = {}
            
            if tags:
                # 提取标签列表
                tag_items = re.findall(r'"(.*?)"', tags[0])
                result["tags"] = tag_items if tag_items else ["未识别"]
            else:
                result["tags"] = ["未识别"]
                
            if scene:
                result["scene"] = scene[0]
            if elements:
                element_items = re.findall(r'"(.*?)"', elements[0])
                result["elements"] = element_items if element_items else []
            if lighting:
                result["lighting"] = lighting[0]
            if mood:
                result["mood"] = mood[0]
                
            result["raw_text"] = text
            return result
            
    def guess_tag_category(self, tag_name: str) -> str:
        """根据标签名称猜测分类"""
        tag_lower = tag_name.lower()
        
        # 场景类别
        if any(word in tag_lower for word in ['室内', '室外', '城市', '自然', '景观', '街道', '办公室', '家']):
            return "场景"
            
        # 光照类别
        if any(word in tag_lower for word in ['白天', '夜晚', '黄昏', '黎明', '阴天', '晴天', '阳光']):
            return "光照"
            
        # 情绪类别
        if any(word in tag_lower for word in ['快乐', '悲伤', '紧张', '兴奋', '平静', '焦虑', '愉快']):
            return "情绪"
            
        # 构图类别
        if any(word in tag_lower for word in ['特写', '中景', '远景', '全景', '俯视', '仰视', '平视']):
            return "构图"
            
        # 主体类别
        if any(word in tag_lower for word in ['人物', '动物', '建筑', '自然', '交通', '食物']):
            return "主体"
            
        # 动作类别
        if any(word in tag_lower for word in ['奔跑', '行走', '交谈', '站立', '坐着', '工作']):
            return "动作"
            
        # 默认类别
        return "其他"