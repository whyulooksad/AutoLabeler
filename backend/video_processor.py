#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频处理模块
实现视频转码和坐标转换功能
"""

import os
import cv2
import numpy as np
import logging
from typing import Tuple, Dict, List, Optional
import tempfile
import shutil
import pickle

logger = logging.getLogger(__name__)

class VideoProcessor:
    """视频处理器，负责视频转码和坐标转换"""
    
    def __init__(self, target_width: int = 960):
        """
        初始化视频处理器
        
        Args:
            target_width: 目标视频宽度，高度按比例自适应
        """
        self.target_width = target_width
        self.temp_dir = "./temp_videos"
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def get_video_info(self, video_path: str) -> Dict:
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            包含视频信息的字典
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                "fps": fps,
                "total_frames": total_frames,
                "original_width": original_width,
                "original_height": original_height,
                "duration": duration,
                "aspect_ratio": original_width / original_height if original_height > 0 else 1
            }
        except Exception as e:
            logger.error(f"获取视频信息失败: {e}")
            raise
    
    def calculate_resize_ratio(self, original_width: int, original_height: int) -> Tuple[float, int, int]:
        """
        计算缩放比例和目标尺寸
        
        Args:
            original_width: 原始宽度
            original_height: 原始高度
            
        Returns:
            (缩放比例, 目标宽度, 目标高度)
        """
        if original_width <= self.target_width:
            # 如果原始宽度已经小于等于目标宽度，不进行缩放
            return 1.0, original_width, original_height
        
        # 计算缩放比例
        scale_ratio = self.target_width / original_width
        target_width = self.target_width
        target_height = int(original_height * scale_ratio)
        
        return scale_ratio, target_width, target_height
    
    def transcode_video(self, original_video_path: str, output_video_path: str = None) -> Tuple[str, Dict]:
        """
        转码视频到较低分辨率
        
        Args:
            original_video_path: 原始视频路径
            output_video_path: 输出视频路径，如果为None则自动生成
            
        Returns:
            (转码后视频路径, 转码信息字典)
        """
        try:
            # 获取原始视频信息
            original_info = self.get_video_info(original_video_path)
            
            # 计算缩放比例和目标尺寸
            scale_ratio, target_width, target_height = self.calculate_resize_ratio(
                original_info["original_width"], 
                original_info["original_height"]
            )
            
            # 生成输出文件路径
            if output_video_path is None:
                base_name = os.path.splitext(os.path.basename(original_video_path))[0]
                output_video_path = os.path.join(
                    self.temp_dir, 
                    f"{base_name}_resized_{target_width}x{target_height}.mp4"
                )
            
            logger.info(f"开始转码视频: {original_video_path}")
            logger.info(f"原始尺寸: {original_info['original_width']}x{original_info['original_height']}")
            logger.info(f"目标尺寸: {target_width}x{target_height}")
            logger.info(f"缩放比例: {scale_ratio:.4f}")
            
            # 打开原始视频
            cap = cv2.VideoCapture(original_video_path)
            
            # 设置输出视频编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_video_path, 
                fourcc, 
                original_info["fps"], 
                (target_width, target_height)
            )
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 调整帧大小
                resized_frame = cv2.resize(frame, (target_width, target_height))
                out.write(resized_frame)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"已处理 {frame_count} 帧")
            
            cap.release()
            out.release()
            
            # 验证转码结果
            resized_info = self.get_video_info(output_video_path)
            
            # 构建转码信息
            transcode_info = {
                "original_video_path": original_video_path,
                "resized_video_path": output_video_path,
                "original_info": original_info,
                "resized_info": resized_info,
                "scale_ratio": scale_ratio,
                "target_width": target_width,
                "target_height": target_height,
                "frame_count": frame_count
            }
            
            logger.info(f"视频转码完成: {output_video_path}")
            logger.info(f"处理帧数: {frame_count}")
            
            return output_video_path, transcode_info
            
        except Exception as e:
            logger.error(f"视频转码失败: {e}")
            raise
    
    def convert_bbox_coordinates(self, 
                                bbox: List[float], 
                                transcode_info: Dict,
                                direction: str = "resized_to_original") -> List[float]:
        """
        转换边界框坐标
        
        Args:
            bbox: 边界框坐标 [x_center, y_center, width, height] (YOLO格式，归一化)
            transcode_info: 转码信息字典
            direction: 转换方向 ("resized_to_original" 或 "original_to_resized")
            
        Returns:
            转换后的边界框坐标
        """
        try:
            # 对于YOLO格式的归一化坐标，由于坐标本身就是相对于图像尺寸的比例
            # 在相同的宽高比下，归一化坐标应该保持不变
            # 只有在宽高比发生变化时才需要调整
            
            original_width = transcode_info["original_info"]["original_width"]
            original_height = transcode_info["original_info"]["original_height"]
            resized_width = transcode_info["resized_info"]["original_width"]
            resized_height = transcode_info["resized_info"]["original_height"]
            
            # 计算宽高比
            original_aspect_ratio = original_width / original_height
            resized_aspect_ratio = resized_width / resized_height
            
            # 如果宽高比相同，坐标保持不变
            if abs(original_aspect_ratio - resized_aspect_ratio) < 1e-6:
                return bbox.copy()
            
            # 转换坐标
            x_center, y_center, width, height = bbox
            
            if direction == "resized_to_original":
                # 从转码视频坐标转换到原始视频坐标
                # 由于转码是按宽度缩放的，高度按比例缩放，所以宽高比可能发生变化
                
                if original_aspect_ratio > resized_aspect_ratio:
                    # 原始视频更宽，需要调整x坐标
                    scale_factor = original_aspect_ratio / resized_aspect_ratio
                    new_x_center = x_center * scale_factor
                    new_width = width * scale_factor
                    new_y_center = y_center
                    new_height = height
                else:
                    # 原始视频更高，需要调整y坐标
                    scale_factor = resized_aspect_ratio / original_aspect_ratio
                    new_y_center = y_center * scale_factor
                    new_height = height * scale_factor
                    new_x_center = x_center
                    new_width = width
                
            elif direction == "original_to_resized":
                # 从原始视频坐标转换到转码视频坐标
                
                if original_aspect_ratio > resized_aspect_ratio:
                    # 原始视频更宽，需要调整x坐标
                    scale_factor = resized_aspect_ratio / original_aspect_ratio
                    new_x_center = x_center * scale_factor
                    new_width = width * scale_factor
                    new_y_center = y_center
                    new_height = height
                else:
                    # 原始视频更高，需要调整y坐标
                    scale_factor = original_aspect_ratio / resized_aspect_ratio
                    new_y_center = y_center * scale_factor
                    new_height = height * scale_factor
                    new_x_center = x_center
                    new_width = width
            else:
                raise ValueError(f"不支持的转换方向: {direction}")
            
            # 确保坐标在有效范围内
            new_x_center = max(0, min(1, new_x_center))
            new_y_center = max(0, min(1, new_y_center))
            new_width = max(0, min(1, new_width))
            new_height = max(0, min(1, new_height))
            
            return [new_x_center, new_y_center, new_width, new_height]
                
        except Exception as e:
            logger.error(f"坐标转换失败: {e}")
            raise
    
    def convert_mask_coordinates(self, 
                                mask: np.ndarray, 
                                transcode_info: Dict,
                                direction: str = "resized_to_original") -> np.ndarray:
        """
        转换mask坐标
        
        Args:
            mask: 二值mask数组
            transcode_info: 转码信息字典
            direction: 转换方向
            
        Returns:
            转换后的mask数组
        """
        try:
            if direction == "resized_to_original":
                # 从转码视频mask转换到原始视频mask
                original_height = transcode_info["original_info"]["original_height"]
                original_width = transcode_info["original_info"]["original_width"]
                resized_height = transcode_info["resized_info"]["original_height"]
                resized_width = transcode_info["resized_info"]["original_width"]
                
                # 调整mask大小
                resized_mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
                return resized_mask
                
            elif direction == "original_to_resized":
                # 从原始视频mask转换到转码视频mask
                original_height = transcode_info["original_info"]["original_height"]
                original_width = transcode_info["original_info"]["original_width"]
                resized_height = transcode_info["resized_info"]["original_height"]
                resized_width = transcode_info["resized_info"]["original_width"]
                
                # 调整mask大小
                resized_mask = cv2.resize(mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
                return resized_mask
            else:
                raise ValueError(f"不支持的转换方向: {direction}")
                
        except Exception as e:
            logger.error(f"Mask坐标转换失败: {e}")
            raise
    
    def preprocess_original_frames(self, transcode_info: Dict, frame_indices: List[int] = None) -> Dict:
        """
        预处理原始视频帧，提高数据集生成效率
        
        Args:
            transcode_info: 转码信息字典
            frame_indices: 需要处理的帧索引列表，如果为None则处理所有帧
            
        Returns:
            包含预处理信息的字典
        """
        try:
            original_video_path = transcode_info["original_video_path"]
            original_info = transcode_info["original_info"]
            
            # 生成缓存文件路径
            base_name = os.path.splitext(os.path.basename(original_video_path))[0]
            cache_dir = os.path.join(self.temp_dir, f"{base_name}_frames_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # 如果frame_indices为None，处理所有帧
            if frame_indices is None:
                frame_indices = list(range(original_info["total_frames"]))
            
            logger.info(f"开始预处理 {len(frame_indices)} 帧原始视频")
            
            # 读取并保存指定帧
            cap = cv2.VideoCapture(original_video_path)
            processed_frames = {}
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # 转换为RGB格式
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 保存帧到缓存文件
                    frame_path = os.path.join(cache_dir, f"frame_{frame_idx:06d}.jpg")
                    cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                    
                    processed_frames[frame_idx] = frame_path
                    
                    if len(processed_frames) % 50 == 0:
                        logger.info(f"已预处理 {len(processed_frames)} 帧")
            
            cap.release()
            
            # 保存预处理信息
            cache_info = {
                "cache_dir": cache_dir,
                "processed_frames": processed_frames,
                "frame_indices": frame_indices,
                "original_info": original_info
            }
            
            cache_file = os.path.join(cache_dir, "cache_info.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_info, f)
            
            logger.info(f"预处理完成，缓存目录: {cache_dir}")
            return cache_info
            
        except Exception as e:
            logger.error(f"预处理原始帧失败: {e}")
            raise
    
    def get_original_frame_efficient(self, frame_idx: int, cache_info: Dict) -> np.ndarray:
        """
        高效获取原始视频帧（从缓存）
        
        Args:
            frame_idx: 帧索引
            cache_info: 缓存信息字典
            
        Returns:
            原始视频帧（RGB格式）
        """
        try:
            if frame_idx not in cache_info["processed_frames"]:
                raise ValueError(f"帧 {frame_idx} 不在缓存中")
            
            frame_path = cache_info["processed_frames"][frame_idx]
            
            # 从缓存文件读取帧
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError(f"无法读取缓存帧: {frame_path}")
            
            # 转换为RGB格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
            
        except Exception as e:
            logger.error(f"获取原始帧失败: {e}")
            raise
    
    def extract_frames_with_sampling(self, video_path: str, sample_interval: int = 8) -> List[np.ndarray]:
        """
        从视频中按指定间隔抽帧
        
        Args:
            video_path: 视频文件路径
            sample_interval: 抽帧间隔，每N帧抽取1帧
            
        Returns:
            抽取的帧列表
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")
            
            frames = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 按间隔抽帧
                if frame_count % sample_interval == 0:
                    # 转换为RGB格式
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"已处理 {frame_count} 帧，抽取 {len(frames)} 帧")
            
            cap.release()
            
            logger.info(f"抽帧完成: 总帧数 {frame_count}，抽取 {len(frames)} 帧，间隔 {sample_interval}")
            return frames
            
        except Exception as e:
            logger.error(f"抽帧失败: {e}")
            raise
    
    def get_sampled_frame_indices(self, total_frames: int, sample_interval: int = 8) -> List[int]:
        """
        获取抽帧的帧索引列表
        
        Args:
            total_frames: 总帧数
            sample_interval: 抽帧间隔
            
        Returns:
            抽帧的帧索引列表
        """
        return list(range(0, total_frames, sample_interval))
    
    def cleanup_temp_files(self, video_path: str = None):
        """
        清理临时文件
        
        Args:
            video_path: 指定要删除的视频文件路径，如果为None则清理所有临时文件
        """
        try:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"已删除临时文件: {video_path}")
            elif video_path is None:
                # 清理整个临时目录
                if os.path.exists(self.temp_dir):
                    shutil.rmtree(self.temp_dir)
                    os.makedirs(self.temp_dir, exist_ok=True)
                    logger.info("已清理所有临时文件")
        except Exception as e:
            logger.error(f"清理临时文件失败: {e}")

# 全局视频处理器实例
video_processor = VideoProcessor()

def get_video_processor() -> VideoProcessor:
    """获取全局视频处理器实例"""
    return video_processor
