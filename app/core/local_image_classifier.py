import os
import json
import time
import base64
import logging
import tempfile
import subprocess
import platform
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from PIL import Image, ImageStat
import io

logger = logging.getLogger(__name__)

class QwenVLClassifier:
    """
    Cross-platform Qwen-VL Image Classifier that properly interfaces with the
    llama-qwen2vl-cli command line tool
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        mmproj_path: Optional[str] = None,
        cli_path: Optional[str] = None,
        model_size: str = "7b",
        max_workers: int = 3,
        custom_tags: Optional[List[str]] = None,
        output_file: Optional[str] = None,
        gpu_layers: int = 0
    ):
        """
        Initialize the Qwen-VL Image Classifier
        
        Args:
            model_path: Path to the Qwen-VL model GGUF file
            mmproj_path: Path to the vision projection file
            cli_path: Path to the llama-qwen2vl-cli executable
            model_size: Model size ("1.5b", "2b", "7b")
            max_workers: Max number of parallel processing threads
            custom_tags: List of custom tags to guide classification
            output_file: Path to save classification results
            gpu_layers: Number of GPU layers to use (0 for CPU-only)
        """
        self.model_size = model_size
        self.model_path = model_path or self._get_default_model_path()
        self.mmproj_path = mmproj_path or self._get_default_mmproj_path()
        self.cli_path = cli_path or self._get_default_cli_path()
        self.max_workers = max_workers
        self.output_file = output_file
        self.custom_tags = custom_tags or []
        self.gpu_layers = gpu_layers
        
        # 尝试从应用配置中获取路径 (如果在应用环境中运行)
        try:
            from app.config import settings
            if not model_path and settings.LOCAL_MODEL_PATH:
                self.model_path = settings.LOCAL_MODEL_PATH
                logger.debug(f"Using model path from app config: {self.model_path}")
            
            if not mmproj_path and settings.QWEN_VL_MMPROJ_PATH:
                self.mmproj_path = settings.QWEN_VL_MMPROJ_PATH
                logger.debug(f"Using mmproj path from app config: {self.mmproj_path}")
                
            if not cli_path and settings.QWEN_VL_CLI_PATH:
                self.cli_path = settings.QWEN_VL_CLI_PATH
                logger.debug(f"Using CLI path from app config: {self.cli_path}")
        except ImportError:
            # 如果不是在应用环境中运行，则忽略
            pass
        
        # 初始化标签存储
        self.tags_data = {}
        if output_file and os.path.exists(output_file):
            self._load_existing_tags()
            
        # 验证路径
        self._validate_paths()
    
    def _get_default_model_path(self) -> str:
        """Get default model path based on platform and model size"""
        home = str(Path.home())
        base_dir = os.path.join(home, "models", "qwen-vl")
        
        # Try different naming patterns
        model_paths = [
            os.path.join(base_dir, f"Qwen2-VL-{self.model_size.upper()}-Instruct-Q4_K_M.gguf"),
            os.path.join(base_dir, f"Qwen2-VL-{self.model_size}-Instruct-Q4_K_M.gguf")
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                return path
        
        # Return the first path as default even if it doesn't exist
        return model_paths[0]
    
    def _get_default_mmproj_path(self) -> str:
        """Get default mmproj path based on platform and model size"""
        home = str(Path.home())
        base_dir = os.path.join(home, "models", "qwen-vl")
        
        # Try different naming patterns for mmproj file
        mmproj_paths = [
            os.path.join(base_dir, f"qwen-qwen2-vl-{self.model_size}-instruct-vision.gguf"),
            os.path.join(base_dir, f"qwen2-vl-{self.model_size}-instruct-vision.gguf")
        ]
        
        for path in mmproj_paths:
            if os.path.exists(path):
                return path
        
        # Return the first path as default even if it doesn't exist
        return mmproj_paths[0]
    
    def _get_default_cli_path(self) -> str:
        """Get default CLI path based on platform"""
        system = platform.system()
        
        if system == "Windows":
            # Check common Windows locations
            candidates = [
                r"C:\llama.cpp\bin\llama-qwen2vl-cli.exe",
                r"C:\Program Files\llama.cpp\bin\llama-qwen2vl-cli.exe",
                os.path.join(str(Path.home()), "llama.cpp", "bin", "llama-qwen2vl-cli.exe")
            ]
        elif system == "Darwin":  # macOS
            candidates = [
                "/usr/local/bin/llama-qwen2vl-cli",
                "/opt/homebrew/bin/llama-qwen2vl-cli",
                os.path.join(str(Path.home()), "llama.cpp", "bin", "llama-qwen2vl-cli")
            ]
        else:  # Linux
            candidates = [
                "/usr/local/bin/llama-qwen2vl-cli",
                "/usr/bin/llama-qwen2vl-cli",
                os.path.join(str(Path.home()), "llama.cpp", "bin", "llama-qwen2vl-cli")
            ]
        
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        
        # Default to just the binary name and hope it's in PATH
        if system == "Windows":
            return "llama-qwen2vl-cli.exe"
        else:
            return "llama-qwen2vl-cli"
    
    def _validate_paths(self) -> None:
        """Validate that all necessary paths exist"""
        missing_files = []
        
        if not os.path.exists(self.model_path):
            missing_files.append(f"Model file: {self.model_path}")
        
        if not os.path.exists(self.mmproj_path):
            missing_files.append(f"Vision projection file: {self.mmproj_path}")
        
        # Check if CLI is in PATH if it's just a binary name
        if not os.path.exists(self.cli_path) and not self.cli_path.startswith("/") and "/" not in self.cli_path and "\\" not in self.cli_path:
            # Try to find the executable in PATH
            from shutil import which
            full_path = which(self.cli_path)
            if full_path:
                self.cli_path = full_path
            else:
                missing_files.append(f"CLI executable: {self.cli_path}")
        
        if missing_files:
            for missing in missing_files:
                logger.warning(f"Missing: {missing}")
            
            logger.warning(
                "Some required files are missing. Please install the Qwen2-VL model properly.\n"
                "1. Download the GGUF model from HuggingFace\n"
                "2. Download the vision projection file\n"
                "3. Install llama.cpp with Qwen2-VL support"
            )
    
    def _load_existing_tags(self) -> None:
        """Load existing tag data from file"""
        if not self.output_file or not os.path.exists(self.output_file):
            return
            
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                self.tags_data = json.load(f)
            logger.info(f"Loaded {len(self.tags_data)} previously tagged images")
        except json.JSONDecodeError:
            logger.warning("Tag file format error, will create new file")
            self.tags_data = {}
    
    def _save_tags(self) -> None:
        """Save tag data to file"""
        if not self.output_file:
            return
            
        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.tags_data, f, ensure_ascii=False, indent=2)
    
    def classify_images(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Batch classify images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary mapping image paths to their tag data
        """
        # Filter out already processed images
        new_images = [img for img in image_paths if img not in self.tags_data]
        if not new_images:
            logger.info("All images already tagged")
            return self.tags_data
        
        logger.info(f"Processing {len(new_images)} new images...")
        
        # Process images with thread pool
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
                        
                        # Save results periodically
                        if results_count % 10 == 0:
                            logger.info(f"Processed {results_count}/{len(new_images)} images")
                            self._save_tags()
                except Exception as e:
                    logger.error(f"Error processing image {img_path}: {e}")
                    # Add fallback result with error information
                    self.tags_data[img_path] = self._fallback_classification(img_path, str(e))
        
        # Save final results
        self._save_tags()
        logger.info(f"Classification complete, processed {results_count} images")
        return self.tags_data
    
    def _prepare_image(self, image_path: str, max_size: int = 640) -> str:
        """
        Prepare image for processing (resize to recommended size)
        
        Args:
            image_path: Path to the image file
            max_size: Maximum dimension size
            
        Returns:
            Path to the prepared image
        """
        try:
            # Check if image needs resizing
            with Image.open(image_path) as img:
                width, height = img.size
                
                # If image is already small enough, use original
                if width <= max_size and height <= max_size:
                    return image_path
                
                # Resize image
                ratio = min(max_size / width, max_size / height)
                new_size = (int(width * ratio), int(height * ratio))
                resized = img.resize(new_size, Image.LANCZOS)
                
                # Save to temporary file
                fd, temp_path = tempfile.mkstemp(suffix='.jpg')
                os.close(fd)
                resized.save(temp_path, format="JPEG", quality=90)
                
                logger.debug(f"Resized image from {width}x{height} to {new_size[0]}x{new_size[1]}")
                return temp_path
        except Exception as e:
            logger.warning(f"Image resizing failed: {e}, using original")
            return image_path
    
    def _classify_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        Classify a single image using llama-qwen2vl-cli
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with classification results
        """
        if not os.path.exists(image_path):
            logger.warning(f"Image does not exist: {image_path}")
            return self._fallback_classification(image_path, "File not found")
        
        try:
            # Prepare prompt with custom tags if available
            custom_tags_hint = ""
            if self.custom_tags:
                custom_tags_hint = f" Consider these specific tags: {', '.join(self.custom_tags)}."
                
            prompt = (
                f"Analyze this image and return a JSON object with the following fields:{custom_tags_hint}\n"
                f"- tags: array of 5-10 relevant keywords describing the content\n"
                f"- scene: type of scene (indoor, outdoor, nature, urban, etc.)\n"
                f"- elements: array of main visual elements in the image\n"
                f"- lighting: lighting conditions (daylight, night, etc.)\n"
                f"- mood: emotional tone or style\n"
                f"- composition: compositional characteristics (close-up, wide shot, etc.)\n\n"
                f"Return ONLY the JSON without explanation."
            )
            
            # Prepare image (resize if needed)
            prepared_image = self._prepare_image(image_path)
            
            # Build command - use list form for subprocess to handle escaping correctly
            cmd = [
                self.cli_path,
                "-m", self.model_path,
                "--mmproj", self.mmproj_path,
                "-p", prompt,  # subprocess will handle proper escaping
                "--image", prepared_image,
                "-ngl", str(self.gpu_layers),
                "-n", "512"  # Max tokens to generate
            ]
            
            logger.debug(f"Running command: {' '.join(cmd)}")
            
            # Execute command using subprocess.Popen with list arguments
            # This avoids shell interpretation issues with the prompt
            start_time = time.time()
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                shell=False  # Important: Don't use shell=True to avoid quoting issues
            )
            stdout, stderr = process.communicate()
            
            # Clean up temporary file if we created one
            if prepared_image != image_path and os.path.exists(prepared_image):
                try:
                    os.unlink(prepared_image)
                except:
                    pass
            
            # Check for errors
            if process.returncode != 0:
                logger.error(f"CLI process error (code {process.returncode}): {stderr}")
                return self._fallback_classification(image_path, f"CLI error: {stderr}")
            
            # Log processing time
            processing_time = time.time() - start_time
            logger.debug(f"Image processed in {processing_time:.2f} seconds")
            
            # Parse output to extract JSON
            result = self._extract_json_from_text(stdout)
            
            # Add source video info
            video_name = os.path.basename(os.path.dirname(image_path))
            result["source_video"] = video_name
            
            # Add processing metadata
            result["processing_time"] = processing_time
            result["processing_method"] = "qwen2vl-cli"
            
            return result
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return self._fallback_classification(image_path, str(e))
    
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON data from the model's output text
        
        Args:
            text: Output text from the model
            
        Returns:
            Parsed JSON data or fallback structure
        """
        import re
        import json
        
        # Try to find JSON block
        json_pattern = r'\{[\s\S]*\}'
        matches = re.findall(json_pattern, text)
        
        if matches:
            # Get the longest match as it's likely the most complete
            json_str = max(matches, key=len)
            
            try:
                # Clean and parse JSON
                json_str = self._clean_json_string(json_str)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.debug(f"Failed JSON string: {json_str}")
        
        # If no valid JSON found, extract data with regex
        return self._extract_data_with_regex(text)
    
    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean a JSON string to make it valid
        
        Args:
            json_str: Raw JSON string
            
        Returns:
            Cleaned JSON string
        """
        import re
        
        # Remove any text before the first {
        start_idx = json_str.find('{')
        if start_idx > 0:
            json_str = json_str[start_idx:]
        
        # Remove any text after the last }
        end_idx = json_str.rfind('}')
        if end_idx >= 0 and end_idx < len(json_str) - 1:
            json_str = json_str[:end_idx+1]
        
        # Replace single quotes with double quotes
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)  # Fix key names
        json_str = re.sub(r':\s*\'([^\']*)\'', r': "\1"', json_str)  # Fix string values
        
        # Remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        return json_str
    
    def _extract_data_with_regex(self, text: str) -> Dict[str, Any]:
        """
        Extract structured data using regex when JSON parsing fails
        
        Args:
            text: Text output from the model
            
        Returns:
            Dictionary with extracted data
        """
        import re
        
        result = {}
        
        # Extract tags
        tags = []
        tag_patterns = [
            r'"tags"\s*:\s*\[(.*?)\]',
            r'"标签"\s*:\s*\[(.*?)\]',
            r'标签[:：]\s*\[(.*?)\]',
            r'标签[:：](.*?)(?:。|；|$)'
        ]
        
        for pattern in tag_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                # Extract quoted items
                tag_items = re.findall(r'"([^"]+)"', matches[0])
                if tag_items:
                    tags.extend(tag_items)
                else:
                    # Try comma-separated items
                    comma_items = [item.strip() for item in matches[0].split(',')]
                    tags.extend([item for item in comma_items if item and not item.isspace()])
                break
        
        # If no tags found, extract keywords
        if not tags:
            keyword_patterns = [
                r'关键词[:：](.*?)(?:。|；|$)',
                r'关键字[:：](.*?)(?:。|；|$)'
            ]
            
            for pattern in keyword_patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    keywords = [k.strip() for k in matches[0].split(',')]
                    tags.extend([k for k in keywords if k and not k.isspace()])
                    break
        
        # If still no tags, look for quoted phrases
        if not tags:
            quote_items = re.findall(r'"([^"]{1,20})"', text)
            tags = [item for item in quote_items if item and not item.isspace()][:5]
        
        # Use default tags if none found
        if not tags:
            tags = ["未识别"]
        
        # Remove duplicates and limit length
        tags = sorted(set(tags), key=len)
        result["tags"] = tags[:10]
        
        # Extract scene, elements, lighting and mood with similar patterns
        for field, patterns in {
            "scene": [r'"scene"\s*:\s*"(.*?)"', r'场景[:：]\s*(.*?)(?:。|；|$)'],
            "elements": [r'"elements"\s*:\s*\[(.*?)\]', r'主要元素[:：]\s*(.*?)(?:。|；|$)'],
            "lighting": [r'"lighting"\s*:\s*"(.*?)"', r'光线[:：]\s*(.*?)(?:。|；|$)'],
            "mood": [r'"mood"\s*:\s*"(.*?)"', r'情绪[:：]\s*(.*?)(?:。|；|$)']
        }.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    if '[' in pattern:
                        # If array format
                        items = re.findall(r'"([^"]+)"', matches[0])
                        if items:
                            result[field] = items
                    else:
                        # If string format
                        value = matches[0].strip()
                        result[field] = value
                    break
            
            # Set default value if not found
            if field not in result:
                if field == "elements":
                    result[field] = []
                else:
                    result[field] = "未识别"
        
        # Add raw text for debugging
        result["raw_text"] = text
        
        return result
    
    def _fallback_classification(self, image_path: str, error_message: str = "") -> Dict[str, Any]:
        """
        Create a fallback classification when the main method fails
        
        Args:
            image_path: Path to the image file
            error_message: Error message if any
            
        Returns:
            Dictionary with basic classification
        """
        # Extract video source information
        video_name = os.path.basename(os.path.dirname(image_path))
        if not video_name or video_name == '.':
            parts = os.path.basename(image_path).split('_')
            video_name = '_'.join(parts[:-1]) if len(parts) > 1 else "unknown_video"
        
        # Get basic image features
        basic_tags = []
        try:
            with Image.open(image_path) as img:
                # Check color mode
                if img.mode == 'RGB':
                    basic_tags.append("彩色")
                elif img.mode == 'L':
                    basic_tags.append("黑白")
                
                # Check orientation
                if img.width > img.height:
                    basic_tags.append("横向")
                else:
                    basic_tags.append("纵向")
                
                # Check brightness
                if img.mode == 'RGB' or img.mode == 'L':
                    stat = ImageStat.Stat(img)
                    brightness = sum(stat.mean) / len(stat.mean)
                    if brightness > 180:
                        basic_tags.append("高亮度")
                    elif brightness < 60:
                        basic_tags.append("低亮度")
                
                # Add timestamp tag from filename
                timestamp_str = os.path.basename(image_path).split('_')[-1].replace('.jpg', '')
                if timestamp_str and len(timestamp_str) >= 6:
                    time_tag = "时间码_" + timestamp_str[:6]
                    basic_tags.append(time_tag)
        except Exception as e:
            logger.error(f"Basic image analysis failed: {e}")
        
        # Create fallback result
        result = {
            "tags": basic_tags if basic_tags else ["自动标记"],
            "source_video": video_name,
            "scene": "未知场景",
            "elements": ["自动生成"],
            "lighting": "未知光照",
            "mood": "未知情绪",
            "method": "fallback_classification",
            "error": error_message
        }
        
        return result
    
    def guess_tag_category(self, tag_name: str) -> str:
        """
        Guess the category of a tag based on its content
        
        Args:
            tag_name: Tag name
            
        Returns:
            Guessed category
        """
        tag_lower = tag_name.lower()
        
        # Scene category
        if any(word in tag_lower for word in ['indoor', 'outdoor', 'city', 'nature', 'landscape', 'street', 'office', 'home', 'urban', 'rural']):
            return "scene"
        
        # Lighting category
        if any(word in tag_lower for word in ['daylight', 'night', 'sunset', 'dawn', 'sunrise', 'cloudy', 'sunny', 'light', 'shadow', 'bright', 'dark']):
            return "lighting"
        
        # Mood category
        if any(word in tag_lower for word in ['happy', 'sad', 'tense', 'excited', 'calm', 'anxious', 'joyful', 'peaceful', 'dramatic', 'serene']):
            return "mood"
        
        # Composition category
        if any(word in tag_lower for word in ['closeup', 'medium shot', 'wide shot', 'panorama', 'overhead', 'low angle', 'high angle', 'eye level']):
            return "composition"
        
        # Subject category
        if any(word in tag_lower for word in ['person', 'people', 'animal', 'building', 'architecture', 'nature', 'transportation', 'food', 'landscape', 'object']):
            return "subject"
        
        # Action category
        if any(word in tag_lower for word in ['running', 'walking', 'talking', 'standing', 'sitting', 'working', 'moving', 'dancing', 'playing']):
            return "action"
        
        # Additional categories
        if any(word in tag_lower for word in ['color', 'black and white', 'monochrome', 'texture', 'pattern']):
            return "style"
        
        # Default category
        return "other"