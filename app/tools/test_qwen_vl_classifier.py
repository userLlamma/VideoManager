#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Qwen-VL Classifier

This script tests the Qwen-VL classifier implementation by classifying a sample image.
It helps verify that the model setup is working correctly.

Usage:
    python test_qwen_vl_classifier.py [path/to/image.jpg]
    
Requirements:
    python-dotenv (optional, for loading .env file)
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to test the Qwen-VL classifier"""
    parser = argparse.ArgumentParser(description="Test Qwen-VL Classifier")
    parser.add_argument("image_path", nargs="?", help="Path to test image")
    parser.add_argument("--model-path", help="Path to model file")
    parser.add_argument("--mmproj-path", help="Path to vision projection file")
    parser.add_argument("--cli-path", help="Path to llama-qwen2vl-cli binary")
    parser.add_argument("--model-size", choices=["2b", "7b"], default="2b", help="Model size")
    parser.add_argument("--gpu-layers", type=int, default=0, help="Number of GPU layers")
    args = parser.parse_args()

    # Try to load .env configuration if no explicit paths provided
    if not (args.model_path and args.mmproj_path):
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            if not args.model_path and os.environ.get('LOCAL_MODEL_PATH'):
                args.model_path = os.environ.get('LOCAL_MODEL_PATH')
                logger.info(f"Using model path from .env: {args.model_path}")
                
            if not args.mmproj_path and os.environ.get('QWEN_VL_MMPROJ_PATH'):
                args.mmproj_path = os.environ.get('QWEN_VL_MMPROJ_PATH')
                logger.info(f"Using mmproj path from .env: {args.mmproj_path}")
                
            if not args.cli_path and os.environ.get('QWEN_VL_CLI_PATH'):
                args.cli_path = os.environ.get('QWEN_VL_CLI_PATH')
                logger.info(f"Using CLI path from .env: {args.cli_path}")
                
            if os.environ.get('LOCAL_MODEL_SIZE'):
                args.model_size = os.environ.get('LOCAL_MODEL_SIZE')
                logger.info(f"Using model size from .env: {args.model_size}")
                
            if os.environ.get('GPU_LAYERS'):
                args.gpu_layers = int(os.environ.get('GPU_LAYERS'))
                logger.info(f"Using GPU layers from .env: {args.gpu_layers}")
        except ImportError:
            logger.warning("dotenv package not installed, can't load .env configuration")
    
    # Import our QwenVLClassifier class
    try:
        # Try to import directly (assuming we're in the project directory)
        try:
            from app.core.qwen_vl_classifier import QwenVLClassifier
            logger.info("Successfully imported QwenVLClassifier from app.core.qwen_vl_classifier")
        except ImportError:
            # If that fails, try to import from current directory
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from qwen_vl_classifier import QwenVLClassifier
            logger.info("Successfully imported QwenVLClassifier from current directory")
        
        # Display configuration
        logger.info("Configuration:")
        logger.info(f"  Model path: {args.model_path}")
        logger.info(f"  Model size: {args.model_size}")
        logger.info(f"  MMPROJ path: {args.mmproj_path}")
        logger.info(f"  CLI path: {args.cli_path}")
        logger.info(f"  GPU layers: {args.gpu_layers}")
        
        # Check if required arguments are provided
        if not args.image_path:
            # Try to find a sample image
            sample_paths = [
                "test_image.jpg",
                "sample.jpg",
                "app/static/sample.jpg",
                os.path.join(os.path.dirname(__file__), "test_image.jpg")
            ]
            
            for path in sample_paths:
                if os.path.exists(path):
                    args.image_path = path
                    logger.info(f"Using sample image: {args.image_path}")
                    break
            
            if not args.image_path:
                logger.error("No image path provided and no sample image found")
                logger.info("Please provide an image path: python test_qwen_vl_classifier.py path/to/image.jpg")
                return 1
        
        # Verify that the image exists
        if not os.path.exists(args.image_path):
            logger.error(f"Image not found: {args.image_path}")
            return 1
        
        # Initialize the classifier
        start_time = time.time()
        logger.info("Initializing QwenVLClassifier...")
        
        classifier = QwenVLClassifier(
            model_path=args.model_path,
            mmproj_path=args.mmproj_path,
            cli_path=args.cli_path,
            model_size=args.model_size,
            max_workers=1,
            gpu_layers=args.gpu_layers
        )
        
        init_time = time.time() - start_time
        logger.info(f"Initialization complete in {init_time:.2f} seconds")
        
        # Classify the image
        logger.info(f"Classifying image: {args.image_path}")
        start_time = time.time()
        
        results = classifier.classify_images([args.image_path])
        
        classification_time = time.time() - start_time
        logger.info(f"Classification complete in {classification_time:.2f} seconds")
        
        # Display results
        if args.image_path in results:
            logger.info("Classification results:")
            result = results[args.image_path]
            
            # Print in a nice format
            print("\n=== Classification Results ===")
            print(f"Source video: {result.get('source_video', 'Unknown')}")
            
            if 'tags' in result:
                print("\nTags:")
                for tag in result['tags']:
                    print(f"  - {tag}")
            
            if 'scene' in result:
                print(f"\nScene: {result['scene']}")
            
            if 'elements' in result:
                print("\nElements:")
                for element in result['elements']:
                    print(f"  - {element}")
            
            if 'lighting' in result:
                print(f"\nLighting: {result['lighting']}")
            
            if 'mood' in result:
                print(f"\nMood: {result['mood']}")
            
            if 'processing_time' in result:
                print(f"\nProcessing time: {result['processing_time']:.2f} seconds")
            
            # Save the full results to a JSON file
            output_file = f"classification_result_{Path(args.image_path).stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved full results to {output_file}")
            
            return 0
        else:
            logger.error("Classification failed: no results returned")
            return 1
        
    except ImportError as e:
        logger.error(f"Error importing QwenVLClassifier: {e}")
        logger.info("Please make sure you have installed all requirements and are running from the project directory")
        return 1
    except Exception as e:
        logger.error(f"Error testing classifier: {e}", exc_info=True)
        return 1