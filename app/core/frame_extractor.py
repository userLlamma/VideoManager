import cv2
import os
import numpy as np
from datetime import timedelta
from typing import List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FrameExtractor:
    """视频帧提取器 - 从视频中提取关键帧"""
    
    def __init__(
        self,
        output_dir: str,
        min_scene_change_threshold: int = 30,
        frame_sample_interval: int = 24,
        quality: int = 95,
        max_frames: int = 200
    ):
        """
        初始化视频帧提取器
        
        参数:
            output_dir: 输出帧的文件夹
            min_scene_change_threshold: 场景变化检测阈值（越低越敏感）
            frame_sample_interval: 帧采样间隔（影响处理速度）
            quality: JPEG保存质量 (1-100)
            max_frames: 每个视频的最大提取帧数
        """
        self.output_dir = output_dir
        self.min_scene_change_threshold = min_scene_change_threshold
        self.frame_sample_interval = frame_sample_interval
        self.quality = quality
        self.max_frames = max_frames
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_frames(self, video_path: str) -> List[Tuple[float, str]]:
        """
        从视频中提取关键帧
        
        参数:
            video_path: 视频文件路径
            
        返回:
            包含(时间码, 文件路径)的提取帧信息列表
        """
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            logger.error(f"视频文件不存在: {video_path}")
            return []
        
        # 打开视频文件
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            logger.error(f"无法打开视频: {video_path}")
            return []
        
        # 获取视频基本信息
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        # 创建视频文件夹
        video_name = Path(video_path).stem
        video_output_dir = os.path.join(self.output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        logger.info(f"处理视频: {video_name}")
        logger.info(f"时长: {timedelta(seconds=duration)}, 帧数: {frame_count}, FPS: {fps}")
        
        # 提取关键帧
        prev_frame = None
        frames_info = []
        frame_index = 0
        extracted_count = 0
        
        # 进度跟踪
        progress_interval = max(1, frame_count // 10)
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
                
            # 只在采样间隔的帧上进行处理
            if frame_index % self.frame_sample_interval == 0:
                # 进度显示
                if frame_index % progress_interval == 0:
                    progress = (frame_index / frame_count) * 100
                    logger.info(f"进度: {progress:.1f}%")
                
                # 转为灰度进行比较
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 第一帧或检测到场景变化
                if prev_frame is None or self._is_scene_change(prev_frame, gray):
                    # 保存帧
                    timestamp = frame_index / fps
                    frame_filename = f"{video_name}_frame_{frame_index}_{self._format_timestamp(timestamp)}.jpg"
                    frame_path = os.path.join(video_output_dir, frame_filename)

                    # 同时存储一个可通过Web访问的路径（用于数据库存储）
                    web_path = os.path.join(self.output_dir, video_name, frame_filename)
                    # 确保路径使用正斜杠，对Windows特别重要
                    web_path = web_path.replace("\\", "/")
                    
                    # 使用JPEG质量设置保存图像
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                    
                    frames_info.append((timestamp, frame_path))
                    extracted_count += 1
                    
                    # 更新比较帧
                    prev_frame = gray
                    
                    # 达到最大帧数限制时停止
                    if extracted_count >= self.max_frames:
                        logger.info(f"已达到最大帧数限制 ({self.max_frames})")
                        break
            
            frame_index += 1
            
        video.release()
        logger.info(f"完成! 共提取 {extracted_count} 帧")
        return frames_info
    
    def _is_scene_change(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> bool:
        """检测两帧之间是否有场景变化"""
        # 计算帧间差异
        frame_diff = cv2.absdiff(prev_frame, curr_frame)
        mean_diff = np.mean(frame_diff)
        return mean_diff > self.min_scene_change_threshold
    
    def _format_timestamp(self, seconds: float) -> str:
        """将秒转换为 HH:MM:SS 格式"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}_{minutes:02d}_{seconds:02d}"