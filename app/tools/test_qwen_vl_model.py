#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：检查Qwen2.5-VL模型的加载和推理功能
用法：python test_qwen_vl_model.py [图像路径] [--model-size 1.5b|7b|72b] [--gpu-layers 0]
"""

import os
import sys
import json
import base64
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_test")

def test_model_loading(
    model_path: Optional[str] = None,
    model_size: str = "7b",
    gpu_layers: int = 0
) -> bool:
    """
    测试模型加载功能
    
    参数:
        model_path: 模型文件路径，如果为None则使用默认路径
        model_size: 模型大小 (1.5b, 7b, 72b)
        gpu_layers: GPU加速的层数
        
    返回:
        加载是否成功
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error("请安装llama-cpp-python: pip install llama-cpp-python")
        return False

    # 如果未指定模型路径，使用默认路径
    if not model_path:
        home = str(Path.home())
        model_filename = f"qwen2_5-vl-{model_size}.Q4_K_M.gguf"
        model_path = os.path.join(home, "models", "qwen-vl", model_filename)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        logger.info(f"请下载Qwen2.5-VL模型并放置于: {model_path}")
        logger.info(f"下载地址参考: https://huggingface.co/Qwen/Qwen2.5-VL-{model_size}-GGUF")
        return False
    
    # 打印模型文件信息
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    logger.info(f"模型文件: {model_path}")
    logger.info(f"文件大小: {file_size_mb:.2f} MB")

    # 设置上下文大小和批处理大小
    if model_size == "1.5b":
        ctx_size = 2048
        batch_size = 512
    elif model_size == "7b":
        ctx_size = 4096
        batch_size = 256
    else:  # 72b
        ctx_size = 8192
        batch_size = 128

    logger.info(f"开始加载模型... (这可能需要几分钟时间)")
    logger.info(f"模型参数: 上下文大小={ctx_size}, 批处理大小={batch_size}, GPU层数={gpu_layers}")
    
    try:
        # 尝试加载模型
        model = Llama(
            model_path=model_path,
            n_ctx=ctx_size,
            n_batch=batch_size,
            n_gpu_layers=gpu_layers,
            verbose=True  # 启用详细日志以便调试
        )
        
        # 检查模型是否已正确加载
        n_vocab = model.n_vocab()
        logger.info(f"模型加载成功: Qwen2.5-VL-{model_size}")
        logger.info(f"词汇表大小: {n_vocab}")
        return True, model
    
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        logger.error(f"错误类型: {type(e).__name__}")
        return False, None

def convert_image_to_base64(image_path: str) -> str:
    """将图像转换为base64格式"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_image_inference(model, image_path: str) -> Dict[str, Any]:
    """
    测试图像推理功能
    
    参数:
        model: 已加载的模型
        image_path: 图像文件路径
        
    返回:
        推理结果
    """
    if not os.path.exists(image_path):
        logger.error(f"图像文件不存在: {image_path}")
        return {"error": "图像文件不存在"}
    
    try:
        # 读取图像并转换为base64
        base64_image = convert_image_to_base64(image_path)
        image_data = f"![image](data:image/jpeg;base64,{base64_image})"
        
        # 创建提示
        prompt = f"""<|im_start|>user
请详细分析这张图像，并提供以下信息:
1. 提供5-10个描述图像内容的标签，每个标签1-3个词
2. 场景类型（如室内、室外、城市、自然等）
3. 主要视觉元素（如人物、建筑、动物等）
4. 时间和光线特征（如白天、夜晚、黄昏等）
5. 情绪或风格（如欢快、严肃、复古等）
请以JSON格式返回结果，包含tags, scene, elements, lighting, mood等字段。

{image_data}
<|im_end|>

<|im_start|>assistant
"""
        
        logger.info("开始图像推理...")
        
        # 使用模型推理
        response = model.create_completion(
            prompt=prompt,
            max_tokens=1024,
            temperature=0.1,
            top_p=0.9,
            stop=["<|im_end|>"]
        )
        
        generated_text = response["choices"][0]["text"]
        logger.info("推理完成，解析结果...")
        
        # 尝试解析JSON
        try:
            # 找到JSON部分
            start_idx = generated_text.find('{')
            end_idx = generated_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_text = generated_text[start_idx:end_idx]
                return json.loads(json_text)
            else:
                return {"raw_text": generated_text}
        except json.JSONDecodeError:
            return {"raw_text": generated_text}
        
    except Exception as e:
        logger.error(f"推理过程出错: {e}")
        return {"error": str(e)}

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试Qwen2.5-VL模型加载和推理")
    parser.add_argument("image_path", nargs="?", help="用于测试的图像文件路径")
    parser.add_argument("--model-path", help="模型文件的完整路径")
    parser.add_argument("--model-size", choices=["1.5b", "7b", "72b"], default="7b", help="模型大小 (默认: 7b)")
    parser.add_argument("--gpu-layers", type=int, default=0, help="GPU加速的层数 (默认: 0, 表示仅CPU)")
    parser.add_argument("--test-load-only", action="store_true", help="仅测试模型加载，不进行推理")
    args = parser.parse_args()
    
    # 打印系统信息
    import platform
    logger.info(f"系统信息: {platform.platform()}")
    logger.info(f"Python版本: {platform.python_version()}")
    
    # 检查llama-cpp-python版本
    try:
        import llama_cpp
        logger.info(f"llama-cpp-python版本: {llama_cpp.__version__}")
    except (ImportError, AttributeError):
        logger.warning("无法获取llama-cpp-python版本")
    
    # 测试模型加载
    logger.info("开始测试模型加载...")
    success, model = test_model_loading(
        model_path=args.model_path,
        model_size=args.model_size,
        gpu_layers=args.gpu_layers
    )
    
    if not success:
        logger.error("模型加载测试失败")
        return 1
    
    logger.info("模型加载测试成功!")
    
    # 如果只测试加载，到此结束
    if args.test_load_only:
        return 0
    
    # 如果没有提供图像路径，退出
    if not args.image_path:
        logger.info("未提供图像路径，测试结束")
        return 0
    
    # 测试图像推理
    logger.info(f"开始测试图像推理: {args.image_path}")
    result = test_image_inference(model, args.image_path)
    
    # 打印结果
    logger.info("推理结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    logger.info("测试完成")
    return 0

if __name__ == "__main__":
    sys.exit(main())