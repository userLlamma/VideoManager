import os
import json
import time
import base64
import logging
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class ImageClassifier:
    """云端图像分类系统 - 支持多种API服务"""
    
    def __init__(
        self, 
        api_type: str = "openai",
        api_key: Optional[str] = None,
        max_workers: int = 5,
        custom_tags: Optional[List[str]] = None,
        output_file: Optional[str] = None
    ):
        """
        初始化云端图像分类器
        
        参数:
            api_type: API类型 ("openai", "aliyun", "azure" 或 "huggingface")
            api_key: API密钥
            max_workers: 并行处理的最大线程数
            custom_tags: 自定义标签列表，用于引导模型生成特定领域的标签
            output_file: 输出的JSON文件路径（可选）
        """
        self.api_type = api_type
        self.api_key = api_key or os.environ.get(f"{api_type.upper()}_API_KEY")
        self.max_workers = max_workers
        self.output_file = output_file
        self.custom_tags = custom_tags or []
        
        # 检查API密钥
        if not self.api_key:
            logger.warning(f"未设置{api_type}的API密钥，请设置{api_type.upper()}_API_KEY环境变量或在初始化时提供")
        
        # 初始化标签存储
        self.tags_data = {}
        if output_file and os.path.exists(output_file):
            self._load_existing_tags()
    
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
        
        # 最终保存结果
        self._save_tags()
        logger.info(f"完成分类，共处理 {results_count} 张图像")
        return self.tags_data
    
    def _classify_single_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        对单个图像进行分类
        
        参数:
            image_path: 图像文件路径
            
        返回:
            图像标签数据
        """
        if not os.path.exists(image_path):
            logger.warning(f"图像不存在: {image_path}")
            return None
        
        # 防止API请求过于频繁
        time.sleep(0.5)
        
        try:
            if self.api_type == "openai":
                return self._classify_with_openai(image_path)
            elif self.api_type == "aliyun":
                return self._classify_with_aliyun(image_path)
            elif self.api_type == "huggingface":
                return self._classify_with_huggingface(image_path)
            elif self.api_type == "azure":
                return self._classify_with_azure(image_path)
            else:
                logger.error(f"不支持的API类型: {self.api_type}")
                return None
        except Exception as e:
            logger.error(f"API调用失败 ({self.api_type}): {e}")
            return None
    
    def _classify_with_openai(self, image_path: str) -> Optional[Dict[str, Any]]:
        """使用OpenAI Vision API分类图像"""
        # 读取图像文件并编码为base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 准备自定义标签提示
        custom_tags_prompt = ""
        if self.custom_tags:
            custom_tags_prompt = f"请从以下标签中选择合适的: {', '.join(self.custom_tags)}。也可以添加不在此列表中但适合的其他标签。"
        
        # 构建API请求
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": "gpt-4-vision-preview",  # 使用Vision模型
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": f"请详细分析这张图像，并为视频素材库提供以下信息:\n"
                                   f"1. 提供5-10个描述图像内容的标签，每个标签1-3个词\n"
                                   f"2. 场景类型（如室内、室外、城市、自然等）\n"
                                   f"3. 主要视觉元素（如人物、建筑、动物等）\n"
                                   f"4. 时间和光线特征（如白天、夜晚、黄昏等）\n"
                                   f"5. 情绪或风格（如欢快、严肃、复古等）\n"
                                   f"{custom_tags_prompt}\n"
                                   f"请以JSON格式返回结果，包含tags, scene, elements, lighting, mood等字段。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        }
        
        # 发送请求
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"OpenAI API错误: {response.status_code} - {response.text}")
            return None
        
        # 解析返回的JSON
        try:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # 尝试从返回的文本中提取JSON部分
            import re
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = content
                
            # 清理非JSON字符并解析
            clean_json_str = re.sub(r'```json|```', '', json_str).strip()
            return json.loads(clean_json_str)
        except Exception as e:
            logger.error(f"解析OpenAI响应出错: {e}")
            logger.debug(f"原始响应: {response.text}")
            return {"tags": ["解析错误"], "raw_response": content}
    
    def _classify_with_aliyun(self, image_path: str) -> Optional[Dict[str, Any]]:
        """使用阿里云视觉智能API分类图像"""
        # 阿里云API需要单独安装SDK
        try:
            from alibabacloud_imageseg20191230.client import Client as ImageSegClient
            from alibabacloud_tea_openapi import models as open_api_models
            from alibabacloud_imageseg20191230 import models as imageseg_models
            from alibabacloud_tea_util import models as util_models
        except ImportError:
            logger.error("请安装阿里云SDK: pip install alibabacloud_imageseg20191230 alibabacloud_tea_openapi")
            return {"tags": ["SDK未安装"]}
        
        # 读取图像并编码
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        
        # 配置阿里云客户端
        config = open_api_models.Config(
            access_key_id=self.api_key,
            access_key_secret=os.environ.get("ALIYUN_SECRET"),
            endpoint="imageseg.cn-shanghai.aliyuncs.com"
        )
        client = ImageSegClient(config)
        
        # 构建请求
        request = imageseg_models.ClassifyImageRequest(
            image_url="data:image/jpeg;base64," + base64_image
        )
        runtime = util_models.RuntimeOptions()
        
        # 发送请求
        try:
            response = client.classify_image_with_options(request, runtime)
            result = response.body.to_map()
            
            # 提取标签
            if "Data" in result and "Tags" in result["Data"]:
                tags = result["Data"]["Tags"]
                return {
                    "tags": [tag["Value"] for tag in tags],
                    "scene": next((tag["Value"] for tag in tags if tag["Type"] == "scene"), "未知"),
                    "elements": [tag["Value"] for tag in tags if tag["Type"] == "object"],
                    "confidence": {tag["Value"]: tag["Confidence"] for tag in tags}
                }
            return {"tags": ["分析失败"], "raw_response": result}
        except Exception as e:
            logger.error(f"阿里云API调用失败: {e}")
            return {"tags": ["API错误"]}
    
    def _classify_with_huggingface(self, image_path: str) -> Optional[Dict[str, Any]]:
        """使用HuggingFace提供的图像分类模型"""
        API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
        
        # 读取图像文件
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        # 发送请求
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(API_URL, headers=headers, data=image_bytes)
        
        if response.status_code != 200:
            logger.error(f"HuggingFace API错误: {response.status_code} - {response.text}")
            return None
        
        # 解析结果
        try:
            result = response.json()
            # 提取标签和置信度
            tags = [{"tag": item["label"], "confidence": item["score"]} for item in result]
            
            # 转换为统一格式
            return {
                "tags": [item["tag"].replace("_", " ") for item in tags],
                "confidence": {item["tag"]: item["confidence"] for item in tags},
                "model": "vit-base-patch16-224"
            }
        except Exception as e:
            logger.error(f"解析HuggingFace响应出错: {e}")
            return {"tags": ["解析错误"]}
    
    def _classify_with_azure(self, image_path: str) -> Optional[Dict[str, Any]]:
        """使用Azure Computer Vision API分类图像"""
        # Azure Computer Vision SDK
        try:
            from azure.cognitiveservices.vision.computervision import ComputerVisionClient
            from msrest.authentication import CognitiveServicesCredentials
        except ImportError:
            logger.error("请安装Azure SDK: pip install azure-cognitiveservices-vision-computervision")
            return {"tags": ["SDK未安装"]}
        
        # 配置Azure客户端
        endpoint = os.environ.get("AZURE_VISION_ENDPOINT")
        if not endpoint:
            logger.error("请设置AZURE_VISION_ENDPOINT环境变量")
            return {"tags": ["配置错误"]}
        
        credentials = CognitiveServicesCredentials(self.api_key)
        client = ComputerVisionClient(endpoint, credentials)
        
        # 读取图像
        with open(image_path, "rb") as image_stream:
            # 分析图像
            features = ['Tags', 'Categories', 'Description', 'Objects', 'Color']
            language = 'zh'
            
            analysis = client.analyze_image_in_stream(
                image_stream, 
                visual_features=features,
                language=language
            )
            
            # 提取结果
            return {
                "tags": [tag.name for tag in analysis.tags],
                "scene": next((category.name for category in analysis.categories), "未知"),
                "description": analysis.description.captions[0].text if analysis.description.captions else "无描述",
                "elements": [obj.object_property for obj in analysis.objects],
                "dominant_colors": analysis.color.dominant_colors,
                "confidence": {tag.name: tag.confidence for tag in analysis.tags}
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