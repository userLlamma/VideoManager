#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced testing script for Qwen2-VL model to debug classification issues
Usage: python test_local_model.py [image_path] [--model-path /path/to/model.gguf] [--gpu-layers 0] [--debug]
"""

import os
import sys
import json
import base64
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("qwen_test.log")
    ]
)
logger = logging.getLogger("qwen_test")

def prepare_image(image_path: str, resize: bool = True, max_size: int = 512) -> str:
    """
    Prepare image for the model - resize and convert to base64
    
    Args:
        image_path: Path to the image file
        resize: Whether to resize the image
        max_size: Maximum dimension size
        
    Returns:
        Base64 encoded image string formatted for model input
    """
    try:
        if resize:
            from PIL import Image
            import io
            
            # Read and resize image
            with Image.open(image_path) as img:
                # Get original dimensions
                original_width, original_height = img.width, img.height
                logger.info(f"Original image dimensions: {original_width}x{original_height}")
                
                # Calculate resize ratio
                ratio = min(max_size / original_width, max_size / original_height)
                new_size = (int(original_width * ratio), int(original_height * ratio))
                
                # Only resize if necessary
                if ratio < 1.0:
                    logger.info(f"Resizing image to {new_size}")
                    img = img.resize(new_size, Image.LANCZOS)
                
                # Convert to RGB mode if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save to buffer
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=90)
                buffer.seek(0)
                
                # Get base64 data
                base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        else:
            # Simply read and encode the image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        return f"![image](data:image/jpeg;base64,{base64_image})"
    except Exception as e:
        logger.error(f"Error preparing image: {e}")
        raise

def test_model_loading(
    model_path: Optional[str] = None,
    gpu_layers: int = 0,
    verbose: bool = False
) -> tuple:
    """
    Test the Qwen2-VL model loading
    
    Args:
        model_path: Path to the model file
        gpu_layers: Number of GPU layers to use
        verbose: Whether to enable verbose logging
        
    Returns:
        Tuple of (success: bool, model: Llama object or None)
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error("Please install llama-cpp-python: pip install llama-cpp-python")
        return False, None

    # If model path not specified, use default path
    if not model_path:
        home = str(Path.home())
        # Try different model files
        model_candidates = [
            os.path.join(home, "models", "qwen-vl", "Qwen2-VL-7B-Instruct-Q4_K_M.gguf"),
            os.path.join(home, "models", "qwen-vl", "Qwen2-VL-2B-Instruct-Q4_K_M.gguf"),
            os.path.join(home, "models", "qwen-vl", "Qwen2-VL-1_5B-Instruct-Q4_K_M.gguf"),
            "/mnt/block_volume/models/qwen-vl/Qwen2-VL-7B-Instruct-Q4_K_M.gguf",
            "/mnt/block_volume/models/qwen-vl/Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
        ]
        
        for candidate in model_candidates:
            if os.path.exists(candidate):
                model_path = candidate
                logger.info(f"Found model at: {model_path}")
                break
        
        if not model_path:
            logger.error("Could not find a Qwen2-VL model file")
            return False, None
    
    # Check if model file exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False, None
    
    # Print model file info
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    logger.info(f"Model file: {model_path}")
    logger.info(f"File size: {file_size_mb:.2f} MB")

    # Set context size based on model filename
    ctx_size = 4096
    batch_size = 256
    
    if "1_5B" in model_path or "1.5B" in model_path:
        ctx_size = 2048
        batch_size = 512
    elif "2B" in model_path:
        ctx_size = 32768
        batch_size = 256
    elif "7B" in model_path:
        ctx_size = 32768
        batch_size = 256
    
    logger.info(f"Loading model... (this may take a few minutes)")
    logger.info(f"Model parameters: context size={ctx_size}, batch size={batch_size}, GPU layers={gpu_layers}")
    
    try:
        # Time the model loading process
        start_time = time.time()
        
        # Load the model
        model = Llama(
            model_path=model_path,
            n_ctx=ctx_size,
            n_batch=batch_size,
            n_gpu_layers=gpu_layers,
            verbose=verbose
        )
        
        # Calculate loading time
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        logger.info(f"Vocabulary size: {model.n_vocab()}")
        
        # Check memory usage
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            logger.info(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
        except ImportError:
            logger.info("psutil not installed, skipping memory usage check")
        
        return True, model
    
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return False, None

def test_prompt_formats(model, image_data: str, debug: bool = False) -> Dict[str, Any]:
    """
    Test different prompt formats to find the most effective one
    
    Args:
        model: Loaded Llama model
        image_data: Base64 encoded image
        debug: Enable debug output
        
    Returns:
        Dictionary with results for different prompt formats
    """
    results = {}
    
    # Define different prompt formats to test
    prompts = [
        # Format 1: Simple prompt with Qwen chat format
        {
            "name": "simple_qwen_format",
            "prompt": f"""<|im_start|>user
Analyze this image and describe what you see.

{image_data}
<|im_end|>

<|im_start|>assistant
""",
            "max_tokens": 256,
        },
        
        # Format 2: JSON format with Qwen chat format
        {
            "name": "json_qwen_format",
            "prompt": f"""<|im_start|>user
Analyze this image and provide tags in JSON format with the following fields:
- "tags": Array of 3-5 descriptive labels
- "scene": Type of scene (indoor, outdoor, etc.)
- "elements": Main visual elements in the image

Return ONLY valid JSON without any explanation.

{image_data}
<|im_end|>

<|im_start|>assistant
```json
""",
            "max_tokens": 512,
            "stop": ["```", "<|im_end|>"]
        },
        
        # Format 3: Structured format without explicitly forcing JSON
        {
            "name": "structured_format",
            "prompt": f"""<|im_start|>user
Analyze this image and answer these questions:
1. What are 3-5 keywords that describe this image?
2. Is this an indoor or outdoor scene?
3. What main objects or elements are visible?

{image_data}
<|im_end|>

<|im_start|>assistant
""",
            "max_tokens": 256,
        }
    ]
    
    # Test each prompt format
    for idx, prompt_config in enumerate(prompts):
        logger.info(f"Testing prompt format {idx+1}/{len(prompts)}: {prompt_config['name']}")
        
        try:
            # Set default parameters if not specified
            max_tokens = prompt_config.get("max_tokens", 512)
            temp = prompt_config.get("temperature", 0.1)
            top_p = prompt_config.get("top_p", 0.9)
            stop = prompt_config.get("stop", ["<|im_end|>"])
            
            # Print prompt in debug mode
            if debug:
                logger.debug(f"Prompt: {prompt_config['prompt']}")
            
            # Time the inference
            start_time = time.time()
            
            # Run inference
            response = model.create_completion(
                prompt=prompt_config['prompt'],
                max_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                stop=stop
            )
            
            # Calculate inference time
            inference_time = time.time() - start_time
            
            # Get generated text
            generated_text = response["choices"][0]["text"] if "choices" in response and response["choices"] else ""
            
            # Store results
            results[prompt_config['name']] = {
                "text": generated_text,
                "inference_time": inference_time
            }
            
            # Try parsing JSON if expected
            if 'json' in prompt_config['name']:
                try:
                    # Clean up the text if needed
                    json_text = generated_text
                    if json_text.strip().endswith('}') and json_text.strip().startswith('{'):
                        parsed_json = json.loads(json_text)
                        results[prompt_config['name']]["parsed_json"] = parsed_json
                    else:
                        results[prompt_config['name']]["json_error"] = "Invalid JSON format"
                except json.JSONDecodeError as e:
                    results[prompt_config['name']]["json_error"] = str(e)
            
            logger.info(f"Inference completed in {inference_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error testing prompt {prompt_config['name']}: {e}")
            results[prompt_config['name']] = {"error": str(e)}
    
    return results

def main():
    """Main function for testing the Qwen2-VL model"""
    parser = argparse.ArgumentParser(description="Test Qwen2-VL model with different prompt formats")
    parser.add_argument("image_path", nargs="?", help="Path to the image file to test")
    parser.add_argument("--model-path", help="Full path to the model file")
    parser.add_argument("--gpu-layers", type=int, default=0, help="Number of GPU layers to use (default: 0, CPU only)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with more verbose output")
    parser.add_argument("--no-resize", action="store_true", help="Don't resize the image before processing")
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Print system information
    import platform
    import subprocess
    
    logger.info(f"System: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    
    # Get CPU info
    try:
        if platform.system() == "Linux":
            cpu_info = subprocess.check_output("lscpu | grep 'Model name'", shell=True).decode().strip()
            logger.info(f"CPU: {cpu_info}")
        elif platform.system() == "Darwin":  # macOS
            cpu_info = subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True).decode().strip()
            logger.info(f"CPU: {cpu_info}")
    except:
        logger.info("Could not retrieve CPU info")
    
    # Check for GPU
    try:
        if platform.system() == "Linux":
            gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True).decode().strip()
            logger.info(f"GPU: {gpu_info}")
    except:
        logger.info("No NVIDIA GPU detected or nvidia-smi not available")
    
    # Check llama-cpp-python version
    try:
        import llama_cpp
        logger.info(f"llama-cpp-python version: {llama_cpp.__version__}")
        
        # Check if CUDA is enabled in llama-cpp
        if hasattr(llama_cpp, '_C'):
            cuda_enabled = getattr(llama_cpp._C, 'LLAMA_SUPPORTS_CUDA', False)
            logger.info(f"CUDA support in llama-cpp: {cuda_enabled}")
    except (ImportError, AttributeError):
        logger.warning("Could not get llama-cpp-python version or CUDA info")
    
    # Test model loading
    logger.info("Starting model loading test...")
    success, model = test_model_loading(
        model_path=args.model_path,
        gpu_layers=args.gpu_layers,
        verbose=args.debug
    )
    
    if not success:
        logger.error("Model loading test failed")
        return 1
    
    logger.info("Model loading test successful!")
    
    # If no image path is provided, exit
    if not args.image_path:
        logger.info("No image path provided, test complete")
        return 0
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        logger.error(f"Image file not found: {args.image_path}")
        return 1
    
    # Process the image
    logger.info(f"Processing image: {args.image_path}")
    image_data = prepare_image(args.image_path, resize=not args.no_resize)
    
    # Test different prompt formats
    logger.info("Testing different prompt formats...")
    results = test_prompt_formats(model, image_data, debug=args.debug)
    
    # Print results
    logger.info("Results of prompt format tests:")
    for name, data in results.items():
        print(f"\n--- Format: {name} ---")
        
        if "error" in data:
            print(f"ERROR: {data['error']}")
            continue
        
        print(f"Inference time: {data['inference_time']:.2f} seconds")
        
        if "json_error" in data:
            print(f"JSON Error: {data['json_error']}")
        
        if "parsed_json" in data:
            print("Parsed JSON:")
            print(json.dumps(data["parsed_json"], indent=2, ensure_ascii=False))
        else:
            print("Generated text:")
            print(data["text"])
    
    # Save complete results to file
    with open("qwen_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info("Test complete. Results saved to qwen_test_results.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())