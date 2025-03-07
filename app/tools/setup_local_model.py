import os
import logging
import argparse
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_file(url, local_path):
    """使用流式下载大文件"""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    try:
        # 尝试使用huggingface_hub直接下载
        logger.info(f"使用HF Hub下载模型...")
        repo_id = url.split('https://huggingface.co/')[1].split('/resolve')[0]
        filename = url.split('main/')[1]
        
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=os.path.dirname(local_path),
            local_dir_use_symlinks=False
        )
        
        if os.path.exists(os.path.join(os.path.dirname(local_path), filename)):
            # 可能下载到了子目录，移动到正确位置
            os.rename(
                os.path.join(os.path.dirname(local_path), filename),
                local_path
            )
            
        logger.info(f"模型下载完成: {local_path}")
        return True
        
    except Exception as e:
        logger.warning(f"HF Hub下载失败: {e}")
        logger.info(f"尝试直接HTTP下载...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # 获取文件大小
            file_size = int(response.headers.get('content-length', 0))
            
            # 检查文件大小 - Qwen2-VL模型应至少为几GB
            if file_size < 1000000:  # 小于1MB肯定有问题
                logger.error(f"下载文件太小 ({file_size} bytes)，可能不是正确的模型文件")
                return False
                
            logger.info(f"文件大小: {file_size/1024/1024:.2f} MB")
            
            with open(local_path, 'wb') as f:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=local_path) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                            
            # 验证文件大小
            downloaded_size = os.path.getsize(local_path)
            if downloaded_size < 1000000:  # 小于1MB
                logger.error(f"下载的模型太小 ({downloaded_size} bytes)，可能不完整")
                return False
                
            logger.info(f"模型下载完成: {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"下载失败: {e}")
            return False

def setup_model(quant, model_dir=None):
    """设置Qwen2-VL-7B-Instruct模型"""
    size = '7b'  # 固定为7B模型
    if quant not in ['Q2_K', 'Q3_K_S', 'Q3_K_M', 'Q4_0', 'Q4_K_M', 'Q5_0', 'Q5_K_M', 'Q6_K', 'Q8_0']:
        logger.error(f"不支持的量化格式: {quant}")
        return False
        
    # 创建模型目录
    if model_dir is None:
        model_dir = os.path.expanduser("~/models/qwen-vl")
    os.makedirs(model_dir, exist_ok=True)
    
    model_filename = f"Qwen2-VL-7B-Instruct-{quant}.gguf"
    model_url = f"https://huggingface.co/bartowski/Qwen2-VL-7B-Instruct-GGUF/resolve/main/{model_filename}"
    model_path = os.path.join(model_dir, model_filename)
    
    logger.info(f"设置Qwen2-VL-7B-Instruct模型，量化格式: {quant}")
    logger.info(f"模型URL: {model_url}")
    logger.info(f"下载位置: {model_path}")
    
    success = download_file(model_url, model_path)
    
    if success:
        file_size_mb = os.path.getsize(model_path) / 1024 / 1024
        logger.info(f"模型大小: {file_size_mb:.2f} MB")
        if file_size_mb < 100:  # 小于100MB可能有问题
            logger.warning(f"模型文件异常小 ({file_size_mb:.2f} MB)，可能不完整")
        
        logger.info("\n设置完成!")
        logger.info(f"模型路径: {model_path}")
        logger.info(f"\n您现在可以更新.env配置文件:")
        logger.info(f"USE_LOCAL_CLASSIFIER=True")
        logger.info(f"LOCAL_MODEL_PATH={model_path}")
        logger.info(f"LOCAL_MODEL_SIZE={size}")
        return True
    else:
        logger.error("模型下载失败，请检查网络连接或尝试手动下载")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="设置Qwen2-VL-7B-Instruct模型")
    parser.add_argument("--quant", type=str, 
                        choices=['Q2_K', 'Q3_K_S', 'Q3_K_M', 'Q4_0', 'Q4_K_M', 'Q5_0', 'Q5_K_M', 'Q6_K', 'Q8_0'],
                        default='Q4_K_M',
                        help="量化格式 (默认: Q4_K_M)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="模型存储目录 (默认: ~/models/qwen-vl)")
    
    args = parser.parse_args()
    setup_model(args.quant, args.model_dir)