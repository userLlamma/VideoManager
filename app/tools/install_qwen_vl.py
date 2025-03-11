#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen-VL Model Installation Script

This script helps users set up the Qwen2-VL model for the video material management system.
It downloads the necessary model files and sets up the configuration.

Usage:
    python install_qwen_vl.py [--model-size 2b|7b] [--install-dir ~/models/qwen-vl]
"""

import os
import sys
import argparse
import logging
import subprocess
import platform
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model URLs
MODEL_URLS = {
    '2b': {
        'model': 'https://huggingface.co/TheBloke/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf',
        'mmproj': 'https://huggingface.co/TheBloke/Qwen2-VL-2B-Instruct-GGUF/resolve/main/qwen-qwen2-vl-2b-instruct-vision.gguf'
    },
    '7b': {
        'model': 'https://huggingface.co/TheBloke/Qwen2-VL-7B-Instruct-GGUF/resolve/main/Qwen2-VL-7B-Instruct-Q4_K_M.gguf',
        'mmproj': 'https://huggingface.co/TheBloke/Qwen2-VL-7B-Instruct-GGUF/resolve/main/qwen-qwen2-vl-7b-instruct-vision.gguf'
    }
}

# Command to check if llama.cpp is installed
LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp.git"

def download_file(url: str, target_path: str) -> bool:
    """
    Download a file from a URL to a target path
    
    Args:
        url: URL to download from
        target_path: Path to save the file to
        
    Returns:
        True if successful, False otherwise
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Determine the download command based on platform
    if shutil.which('wget'):
        cmd = ['wget', '-q', '--show-progress', url, '-O', target_path]
    elif shutil.which('curl'):
        cmd = ['curl', '-L', '-o', target_path, url, '--progress-bar']
    else:
        logger.error("Neither wget nor curl is available. Please install one of them.")
        return False
    
    logger.info(f"Downloading {url}")
    logger.info(f"Saving to {target_path}")
    
    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            logger.info(f"Downloaded successfully")
            
            # Check file size to ensure it's not too small
            file_size = os.path.getsize(target_path)
            if file_size < 1000000:  # Less than 1MB
                logger.error(f"Downloaded file is too small ({file_size} bytes), likely incomplete")
                return False
                
            return True
        else:
            logger.error(f"Download failed with return code {result.returncode}")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Error during download: {e}")
        return False

def check_install_llama_cpp() -> Tuple[bool, Optional[str]]:
    """
    Check if llama.cpp is installed and build it if necessary
    
    Returns:
        Tuple of (success, binary_path)
    """
    logger.info("Checking for llama.cpp...")
    
    # Check if the binary is in PATH
    binary_name = "llama-qwen2vl-cli" + (".exe" if platform.system() == "Windows" else "")
    binary_path = shutil.which(binary_name)
    
    if binary_path:
        logger.info(f"Found llama-qwen2vl-cli at {binary_path}")
        return True, binary_path
    
    # Check common installation locations
    if platform.system() == "Windows":
        common_paths = [
            r"C:\llama.cpp\bin\llama-qwen2vl-cli.exe",
            r"C:\Program Files\llama.cpp\bin\llama-qwen2vl-cli.exe",
            os.path.join(str(Path.home()), "llama.cpp", "bin", "llama-qwen2vl-cli.exe")
        ]
    else:  # Linux/macOS
        common_paths = [
            "/usr/local/bin/llama-qwen2vl-cli",
            "/usr/bin/llama-qwen2vl-cli",
            os.path.join(str(Path.home()), "llama.cpp", "bin", "llama-qwen2vl-cli")
        ]
    
    for path in common_paths:
        if os.path.exists(path):
            logger.info(f"Found llama-qwen2vl-cli at {path}")
            return True, path
    
    # Suggest installation steps
    logger.warning("llama-qwen2vl-cli not found. You need to install llama.cpp with Qwen-VL support.")
    logger.info("Installation instructions:")
    logger.info("1. Clone the repository: git clone https://github.com/ggerganov/llama.cpp.git")
    logger.info("2. Build with Qwen-VL support: cd llama.cpp && mkdir build && cd build && cmake .. -DLLAMA_QWEN2VL=ON && cmake --build . --config Release")
    
    # Ask user if they want to install automatically
    answer = input("Would you like to attempt automatic installation? (y/n): ")
    
    if answer.lower() not in ['y', 'yes']:
        logger.info("Please install llama.cpp manually and run this script again.")
        return False, None
    
    # Try to install automatically
    try:
        # Clone repository
        repo_dir = os.path.join(str(Path.home()), "llama.cpp")
        if not os.path.exists(repo_dir):
            logger.info(f"Cloning llama.cpp repository to {repo_dir}...")
            subprocess.run(["git", "clone", LLAMA_CPP_REPO, repo_dir], check=True)
        else:
            logger.info(f"llama.cpp repository already exists at {repo_dir}")
            logger.info("Updating repository...")
            subprocess.run(["git", "-C", repo_dir, "pull"], check=True)
        
        # Build with Qwen-VL support
        build_dir = os.path.join(repo_dir, "build")
        os.makedirs(build_dir, exist_ok=True)
        
        # Run cmake
        logger.info("Configuring build with Qwen-VL support...")
        subprocess.run(["cmake", "..", "-DLLAMA_QWEN2VL=ON"], cwd=build_dir, check=True)
        
        # Build
        logger.info("Building llama.cpp (this may take a few minutes)...")
        subprocess.run(["cmake", "--build", ".", "--config", "Release"], cwd=build_dir, check=True)
        
        # Check if binary was created
        if platform.system() == "Windows":
            binary_path = os.path.join(build_dir, "bin", "Release", "llama-qwen2vl-cli.exe")
        else:
            binary_path = os.path.join(build_dir, "bin", "llama-qwen2vl-cli")
        
        if os.path.exists(binary_path):
            logger.info(f"Successfully built llama-qwen2vl-cli at {binary_path}")
            return True, binary_path
        else:
            logger.error(f"Build completed but binary not found at {binary_path}")
            return False, None
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Build failed: {e}")
        return False, None
    except Exception as e:
        logger.error(f"Error during installation: {e}")
        return False, None

def download_models(model_size: str, install_dir: str) -> bool:
    """
    Download the Qwen-VL model files
    
    Args:
        model_size: Model size ('2b' or '7b')
        install_dir: Directory to install models to
        
    Returns:
        True if successful, False otherwise
    """
    if model_size not in MODEL_URLS:
        logger.error(f"Invalid model size: {model_size}. Must be one of: {', '.join(MODEL_URLS.keys())}")
        return False
    
    # Create install directory
    os.makedirs(install_dir, exist_ok=True)
    
    # Download model file
    model_url = MODEL_URLS[model_size]['model']
    model_filename = os.path.basename(model_url)
    model_path = os.path.join(install_dir, model_filename)
    
    # Download mmproj file
    mmproj_url = MODEL_URLS[model_size]['mmproj']
    mmproj_filename = os.path.basename(mmproj_url)
    mmproj_path = os.path.join(install_dir, mmproj_filename)
    
    # Download both files
    logger.info(f"Downloading Qwen2-VL-{model_size}-Instruct model...")
    if not download_file(model_url, model_path):
        return False
    
    logger.info(f"Downloading Qwen2-VL-{model_size}-Instruct vision projection file...")
    if not download_file(mmproj_url, mmproj_path):
        return False
    
    logger.info(f"Model files downloaded successfully to {install_dir}")
    return True

def update_config(model_path: str, mmproj_path: str, model_size: str, binary_path: Optional[str]) -> bool:
    """
    Update the application configuration
    
    Args:
        model_path: Path to the model file
        mmproj_path: Path to the mmproj file
        model_size: Model size ('2b' or '7b')
        binary_path: Path to the llama-qwen2vl-cli binary
        
    Returns:
        True if successful, False otherwise
    """
    success = True
    
    # 1. Update .env file
    env_path = '.env'
    if not os.path.exists(env_path):
        # Try to create it
        try:
            with open(env_path, 'w') as f:
                f.write("# Environment configuration for video material management system\n\n")
        except Exception as e:
            logger.error(f"Failed to create .env file: {e}")
            success = False
    
    if os.path.exists(env_path):
        # Read current .env file
        try:
            with open(env_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Failed to read .env file: {e}")
            lines = []
            success = False
        
        # Modify settings
        new_lines = []
        settings_updated = {
            'USE_LOCAL_CLASSIFIER': False,
            'LOCAL_MODEL_PATH': False,
            'LOCAL_MODEL_SIZE': False,
            'QWEN_VL_MMPROJ_PATH': False,
            'QWEN_VL_CLI_PATH': False,
            'GPU_LAYERS': False
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith('USE_LOCAL_CLASSIFIER='):
                new_lines.append('USE_LOCAL_CLASSIFIER=True')
                settings_updated['USE_LOCAL_CLASSIFIER'] = True
            elif line.startswith('LOCAL_MODEL_PATH='):
                new_lines.append(f'LOCAL_MODEL_PATH={model_path}')
                settings_updated['LOCAL_MODEL_PATH'] = True
            elif line.startswith('LOCAL_MODEL_SIZE='):
                new_lines.append(f'LOCAL_MODEL_SIZE={model_size}')
                settings_updated['LOCAL_MODEL_SIZE'] = True
            elif line.startswith('QWEN_VL_MMPROJ_PATH='):
                new_lines.append(f'QWEN_VL_MMPROJ_PATH={mmproj_path}')
                settings_updated['QWEN_VL_MMPROJ_PATH'] = True
            elif line.startswith('QWEN_VL_CLI_PATH=') and binary_path:
                new_lines.append(f'QWEN_VL_CLI_PATH={binary_path}')
                settings_updated['QWEN_VL_CLI_PATH'] = True
            elif line.startswith('GPU_LAYERS='):
                new_lines.append('GPU_LAYERS=0')
                settings_updated['GPU_LAYERS'] = True
            else:
                new_lines.append(line)
        
        # Add settings that weren't updated
        if not settings_updated['USE_LOCAL_CLASSIFIER']:
            new_lines.append('USE_LOCAL_CLASSIFIER=True')
        if not settings_updated['LOCAL_MODEL_PATH']:
            new_lines.append(f'LOCAL_MODEL_PATH={model_path}')
        if not settings_updated['LOCAL_MODEL_SIZE']:
            new_lines.append(f'LOCAL_MODEL_SIZE={model_size}')
        if not settings_updated['QWEN_VL_MMPROJ_PATH']:
            new_lines.append(f'QWEN_VL_MMPROJ_PATH={mmproj_path}')
        if binary_path and not settings_updated['QWEN_VL_CLI_PATH']:
            new_lines.append(f'QWEN_VL_CLI_PATH={binary_path}')
        if not settings_updated['GPU_LAYERS']:
            new_lines.append('GPU_LAYERS=0')
        
        # Write updated .env file
        try:
            with open(env_path, 'w') as f:
                for line in new_lines:
                    f.write(line + '\n')
            logger.info(f"Configuration updated in {env_path}")
        except Exception as e:
            logger.error(f"Failed to update .env file: {e}")
            success = False
    
    # 2. Update config.json file
    config_path = 'config.json'
    if os.path.exists(config_path):
        try:
            # Read current config
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Update local_classifier section
            if 'local_classifier' not in config:
                config['local_classifier'] = {}
            
            config['local_classifier']['use_local_classifier'] = True
            config['local_classifier']['model_path'] = model_path
            config['local_classifier']['model_size'] = model_size
            config['local_classifier']['mmproj_path'] = mmproj_path
            if binary_path:
                config['local_classifier']['cli_path'] = binary_path
            config['local_classifier']['gpu_layers'] = 0
            
            # Write updated config
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Configuration updated in {config_path}")
        except Exception as e:
            logger.error(f"Failed to update config.json: {e}")
            success = False
    else:
        # Create new config.json file
        try:
            config = {
                'local_classifier': {
                    'use_local_classifier': True,
                    'model_path': model_path,
                    'model_size': model_size,
                    'mmproj_path': mmproj_path,
                    'gpu_layers': 0
                }
            }
            
            if binary_path:
                config['local_classifier']['cli_path'] = binary_path
                
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Created new configuration file {config_path}")
        except Exception as e:
            logger.error(f"Failed to create config.json: {e}")
            success = False
    
    return success

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Qwen-VL Model Installation Script")
    parser.add_argument("--model-size", choices=['2b', '7b'], default='2b',
                      help="Model size to install (default: 2b)")
    parser.add_argument("--install-dir", type=str, 
                      help="Directory to install models to (default: ~/models/qwen-vl)")
    args = parser.parse_args()
    
    # Set default install directory if not specified
    if not args.install_dir:
        args.install_dir = os.path.join(str(Path.home()), "models", "qwen-vl")
    
    # Print system information
    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Python version: {platform.python_version()}")
    
    # Check for llama.cpp
    llama_cpp_installed, binary_path = check_install_llama_cpp()
    if not llama_cpp_installed:
        logger.warning("Proceeding without llama.cpp, but you will need to install it later")
    
    # Download models
    if not download_models(args.model_size, args.install_dir):
        logger.error("Failed to download models")
        return 1
    
    # Get full paths to model files
    model_path = os.path.join(args.install_dir, f"Qwen2-VL-{args.model_size.upper()}-Instruct-Q4_K_M.gguf")
    mmproj_path = os.path.join(args.install_dir, f"qwen-qwen2-vl-{args.model_size}-instruct-vision.gguf")
    
    # Update configuration
    if not update_config(model_path, mmproj_path, args.model_size, binary_path):
        logger.error("Failed to update configuration")
        return 1
    
    # Print success message
    logger.info("\nInstallation complete!")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Vision projection path: {mmproj_path}")
    
    if binary_path:
        logger.info(f"llama-qwen2vl-cli path: {binary_path}")
    else:
        logger.warning("\nYou still need to install llama.cpp with Qwen-VL support.")
        logger.info("Installation instructions:")
        logger.info("1. Clone the repository: git clone https://github.com/ggerganov/llama.cpp.git")
        logger.info("2. Build with Qwen-VL support: cd llama.cpp && mkdir build && cd build && cmake .. -DLLAMA_QWEN2VL=ON && cmake --build . --config Release")
    
    logger.info("\nYou can now run the application with local Qwen-VL classification.")
    return 0

if __name__ == "__main__":
    sys.exit(main())