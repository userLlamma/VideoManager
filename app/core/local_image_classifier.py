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
        对单个图像进行分类 - 使用更简短和结构化的提示，增强错误处理
        
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
            logger.debug(f"准备处理图像: {os.path.basename(image_path)}")
            
            # 更结构化的提示，强制要求JSON格式
            prompt = f"""<|im_start|>user
    分析图像并以JSON格式返回以下信息:
    1. tags: 3-5个描述内容的简短标签
    2. scene: 场景类型（如室内、室外等）
    3. elements: 主要视觉元素列表

    格式示例:
    ```json
    {{
    "tags": ["标签1", "标签2", "标签3"],
    "scene": "场景描述",
    "elements": ["元素1", "元素2"]
    }}
    ```

    请确保返回有效的JSON格式，不要有任何额外的解释。

    {image_data}
    <|im_end|>

    <|im_start|>assistant
    ```json
    """

            try:
                # 使用模型推理
                logger.debug("开始模型推理...")
                response = self.model.create_completion(
                    prompt=prompt,
                    max_tokens=512,
                    temperature=0.1,
                    top_p=0.9,
                    stop=["<|im_end|>", "```"]
                )
                
                # 获取生成的文本
                if not response or "choices" not in response or not response["choices"]:
                    logger.error("模型返回空响应或格式不正确")
                    return self._fallback_classification(image_path)
                    
                generated_text = response["choices"][0]["text"]
                
                # 记录原始输出
                logger.debug(f"模型原始输出前100个字符: {generated_text[:100] if generated_text else 'None'}")
                
                if not generated_text or generated_text.isspace():
                    logger.warning("模型返回空文本，使用回退分类")
                    return self._fallback_classification(image_path)
                
                # 确保文本以}结尾，修复可能的不完整JSON
                if not generated_text.strip().endswith("}"):
                    logger.debug("JSON不完整，尝试修复")
                    if "{" in generated_text and not "}" in generated_text:
                        generated_text = generated_text.strip() + "}"
                        logger.debug("添加缺失的右大括号")
                        
                # 解析JSON
                logger.debug("开始解析JSON")
                result = self._extract_json_from_text(generated_text)
                
                # 检查结果是否有基本字段
                if not isinstance(result, dict) or "tags" not in result or not result["tags"]:
                    logger.warning("解析结果不包含有效标签，尝试再次解析")
                    # 再次尝试直接从文本中提取
                    result = self._extract_data_with_regex(generated_text)
                
                # 提取视频源信息并添加到结果中
                video_name = os.path.basename(os.path.dirname(image_path))
                if not video_name or video_name == '.':
                    # 如果无法从路径获取，使用图像文件名的一部分
                    parts = os.path.basename(image_path).split('_')
                    video_name = '_'.join(parts[:-1]) if len(parts) > 1 else "unknown_video"
                    
                # 确保结果是字典类型并添加source_video属性
                if not isinstance(result, dict):
                    logger.error("结果不是字典类型，创建默认字典")
                    result = {"tags": ["无法解析"], "error": "模型输出格式不正确"}
                
                result["source_video"] = video_name
                
                # 记录最终结果中的标签
                logger.info(f"图像 {os.path.basename(image_path)} 的标签: {result.get('tags', [])}")
                return result
                
            except Exception as e:
                logger.error(f"模型推理失败: {e}")
                logger.debug(f"错误类型: {type(e).__name__}")
                # 使用回退机制
                return self._fallback_classification(image_path)
                
        except Exception as e:
            logger.error(f"分类失败: {e}")
            logger.debug(f"错误类型: {type(e).__name__}")
            # 即使在完全失败时也返回带有source_video的结果
            return {
                "tags": ["处理错误"],
                "source_video": os.path.basename(os.path.dirname(image_path)),
                "error": str(e)
            }
            
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """从生成的文本中提取JSON信息 - 增强版，支持多种格式"""
        import re
        import json
        
        # Step 1: Try to find JSON block with code markers
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        json_match = re.search(json_pattern, text)
        
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Step 2: Try to find JSON-like structure using regex
            # Look for {...} including all content inside
            curly_pattern = r'\{[\s\S]*?\}'
            curly_match = re.search(curly_pattern, text)
            
            if curly_match:
                json_str = curly_match.group(0).strip()
            else:
                # Step 3: Use the whole text as a fallback
                json_str = text
        
        # Try to parse the found JSON string
        try:
            # Clean the string - removes non-JSON artifacts
            json_str = self._clean_json_string(json_str)
            
            # Parse JSON
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            
            # Step 4: If JSON parsing fails, extract data with regex
            return self._extract_data_with_regex(text)
    
    def _clean_json_string(self, json_str: str) -> str:
        """
        清理JSON字符串，移除各种干扰字符
        """
        import re
        
        # Remove any prefixes before opening brace
        start_idx = json_str.find('{')
        if start_idx > 0:
            json_str = json_str[start_idx:]
        
        # Remove any suffixes after closing brace
        end_idx = json_str.rfind('}')
        if end_idx >= 0 and end_idx < len(json_str) - 1:
            json_str = json_str[:end_idx+1]
        
        # Fix common formatting issues
        # Replace single quotes with double quotes where appropriate
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)  # Fix key names
        json_str = re.sub(r':\s*\'([^\']*)\'', r': "\1"', json_str)  # Fix string values
        
        # Remove trailing commas before closing brackets (invalid JSON)
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Remove control characters that might break JSON
        json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
        
        return json_str

    def _extract_data_with_regex(self, text: str) -> Dict[str, Any]:
        """
        使用正则表达式从文本中提取结构化数据
        当JSON解析失败时使用
        """
        import re
        
        result = {}
        
        # 使用正则表达式提取各种字段
        
        # 提取标签
        tags = []
        # 尝试找到标签列表
        tag_patterns = [
            r'"tags"\s*:\s*\[(.*?)\]',  # "tags": [...]
            r'"标签"\s*:\s*\[(.*?)\]',  # "标签": [...]
            r'标签[:：]\s*\[(.*?)\]',   # 标签: [...]
            r'标签[:：](.*?)(?:。|；|$)',  # 标签: ... 后跟句号或分号
            r'标签[:：](.*?)(?:\n|\r|$)'   # 标签: ... 后跟换行
        ]
        
        for pattern in tag_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                # 提取引号内的内容
                tag_items = re.findall(r'"([^"]+)"', matches[0])
                if tag_items:
                    tags.extend(tag_items)
                else:
                    # 尝试提取逗号分隔的项
                    comma_items = [item.strip() for item in matches[0].split(',')]
                    tags.extend([item for item in comma_items if item and not item.isspace()])
                break
        
        # 如果没有找到标签，尝试从文本中提取关键词
        if not tags:
            # 寻找关键词部分
            keyword_patterns = [
                r'关键词[:：](.*?)(?:。|；|$)',
                r'关键字[:：](.*?)(?:。|；|$)',
                r'标签[:：](.*?)(?:。|；|$)'
            ]
            
            for pattern in keyword_patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    # 分割逗号分隔的关键词
                    keywords = [k.strip() for k in matches[0].split(',')]
                    tags.extend([k for k in keywords if k and not k.isspace()])
                    break
        
        # 如果仍然没有标签，尝试提取所有引号内的短语作为可能的标签
        if not tags:
            quote_items = re.findall(r'"([^"]{1,20})"', text)  # 限制长度为1-20个字符
            tags = [item for item in quote_items if item and not item.isspace()][:5]  # 最多取5个
        
        # 如果还是没有标签，使用默认标签
        if not tags:
            tags = ["未识别"]
        
        # 去除重复并优先保留短标签
        tags = sorted(set(tags), key=len)
        result["tags"] = tags[:10]  # 限制最多10个标签
        
        # 提取场景
        scene_patterns = [
            r'"scene"\s*:\s*"(.*?)"',
            r'"场景"\s*:\s*"(.*?)"',
            r'场景[:：]\s*(.*?)(?:。|；|$)',
            r'场景类型[:：]\s*(.*?)(?:。|；|$)'
        ]
        
        for pattern in scene_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                result["scene"] = matches[0].strip()
                break
        
        # 提取主要内容/元素
        element_patterns = [
            r'"elements"\s*:\s*\[(.*?)\]',
            r'"主要内容"\s*:\s*"(.*?)"',
            r'主要内容[:：]\s*(.*?)(?:。|；|$)',
            r'主要元素[:：]\s*(.*?)(?:。|；|$)',
            r'视觉元素[:：]\s*(.*?)(?:。|；|$)'
        ]
        
        for pattern in element_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                if '[' in pattern:
                    # 如果是数组格式
                    items = re.findall(r'"([^"]+)"', matches[0])
                    if items:
                        result["elements"] = items
                else:
                    # 如果是字符串格式
                    elements = [e.strip() for e in matches[0].split(',')]
                    result["elements"] = [e for e in elements if e and not e.isspace()]
                break
        
        # 提取光线/时间特征
        lighting_patterns = [
            r'"lighting"\s*:\s*"(.*?)"',
            r'"光线"\s*:\s*"(.*?)"',
            r'光线[:：]\s*(.*?)(?:。|；|$)',
            r'时间[:：]\s*(.*?)(?:。|；|$)'
        ]
        
        for pattern in lighting_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                result["lighting"] = matches[0].strip()
                break
        
        # 添加原始文本以便调试
        result["raw_text"] = text
        
        return result

    def _fallback_classification(self, image_path: str) -> Dict[str, Any]:
        """
        当模型分类失败时的简单回退分类方法
        基于文件名和基本图像特征进行简单分类
        """
        try:
            # 从路径获取视频源信息
            video_name = os.path.basename(os.path.dirname(image_path))
            if not video_name or video_name == '.':
                parts = os.path.basename(image_path).split('_')
                video_name = '_'.join(parts[:-1]) if len(parts) > 1 else "unknown_video"
            
            # 尝试使用PIL获取基本图像信息
            from PIL import Image, ImageStat
            import io
            
            basic_tags = []
            try:
                with Image.open(image_path) as img:
                    # 检查是否为彩色图像
                    if img.mode == 'RGB':
                        basic_tags.append("彩色")
                    elif img.mode == 'L':
                        basic_tags.append("黑白")
                    
                    # 检查图像尺寸
                    if img.width > img.height:
                        basic_tags.append("横向")
                    else:
                        basic_tags.append("纵向")
                    
                    # 检查亮度
                    if img.mode == 'RGB' or img.mode == 'L':
                        stat = ImageStat.Stat(img)
                        brightness = sum(stat.mean) / len(stat.mean)
                        if brightness > 180:
                            basic_tags.append("高亮度")
                        elif brightness < 60:
                            basic_tags.append("低亮度")
                    
                    # 根据时间戳添加标签
                    timestamp_str = os.path.basename(image_path).split('_')[-1].replace('.jpg', '')
                    if timestamp_str and len(timestamp_str) >= 6:
                        time_tag = "时间码_" + timestamp_str[:6]
                        basic_tags.append(time_tag)
                    
                    # 添加帧号信息
                    frame_parts = os.path.basename(image_path).split('_')
                    if len(frame_parts) > 1 and frame_parts[1].startswith('frame'):
                        basic_tags.append(f"帧{frame_parts[2]}")
                    
            except Exception as e:
                logger.error(f"基本图像分析失败: {e}")
            
            # 添加预设标签
            scene_tags = ["室内", "室外", "日景", "夜景"]
            element_tags = ["人物", "场景", "物体"]
            
            # 创建回退结果
            return {
                "tags": basic_tags if basic_tags else ["自动标记"],
                "source_video": video_name,
                "scene": scene_tags[hash(image_path) % len(scene_tags)],  # 随机分配一个场景标签
                "elements": [element_tags[hash(image_path) % len(element_tags)]],  # 随机分配一个元素标签
                "method": "回退分类"
            }
        except Exception as e:
            logger.error(f"回退分类失败: {e}")
            return {
                "tags": ["处理错误"],
                "source_video": os.path.basename(os.path.dirname(image_path)),
                "error": str(e)
            }
    
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