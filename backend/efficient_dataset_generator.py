#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高效数据集生成器
支持批量处理和并行处理，大幅提升数据集生成效率
"""

import os
import cv2
import numpy as np
import logging
import time
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from backend.video_processor import VideoProcessor
import PIL.Image

logger = logging.getLogger(__name__)

class EfficientDatasetGenerator:
    """高效数据集生成器"""
    
    def __init__(self, max_workers: int = None):
        """
        初始化数据集生成器
        
        Args:
            max_workers: 最大工作线程数，默认为CPU核心数
        """
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.video_processor = VideoProcessor()
        
    def generate_yolo_dataset_efficient(self, 
                                      video_state: Dict, 
                                      interactive_state: Dict, 
                                      mask_dropdown: List[str],
                                      output_dir: str = None) -> Tuple[str, List]:
        """
        高效生成YOLO数据集
        
        Args:
            video_state: 视频状态字典
            interactive_state: 交互状态字典
            mask_dropdown: 选择的mask列表
            output_dir: 输出目录
            
        Returns:
            (输出目录路径, 操作日志列表)
        """
        operation_log = [("", ""), ("开始高效生成YOLO数据集...", "Normal")]
        
        try:
            start_time = time.time()
            
            # 验证输入数据
            if not video_state or not video_state.get("origin_images"):
                raise ValueError("没有可用的视频数据")
            
            frames = video_state["origin_images"]
            masks = video_state["masks"]
            
            # 确定mask编号
            mask_numbers = self._get_mask_numbers(mask_dropdown, masks)
            
            # 确定处理范围
            start_frame, end_frame = self._get_processing_range(video_state, interactive_state, len(frames))
            
            # 创建输出目录
            if output_dir is None:
                output_dir = f"./temp_yolo_datasets/yolo_dataset_{int(time.time())}"
            
            images_dir, labels_dir = self._create_output_directories(output_dir)
            
            # 检查是否需要坐标转换
            transcode_info = video_state.get("transcode_info")
            need_coordinate_conversion = transcode_info is not None
            
            # 预处理原始帧（如果需要坐标转换）
            cache_info = None
            if need_coordinate_conversion:
                logger.info("🔄 检测到视频转码，开始高效预处理...")
                frame_indices = list(range(start_frame, end_frame))
                cache_info = self.video_processor.preprocess_original_frames(transcode_info, frame_indices)
                logger.info(f"✅ 预处理完成，缓存了 {len(frame_indices)} 帧")
            
            # 批量处理帧
            logger.info(f"🚀 开始批量处理 {end_frame - start_frame} 帧，使用 {self.max_workers} 个线程")
            
            # 准备批量处理任务
            tasks = []
            for frame_idx in range(start_frame, end_frame):
                if frame_idx < len(frames) and frame_idx < len(masks):
                    task = {
                        'frame_idx': frame_idx,
                        'frame': frames[frame_idx],
                        'mask': masks[frame_idx],
                        'mask_numbers': mask_numbers,
                        'need_coordinate_conversion': need_coordinate_conversion,
                        'transcode_info': transcode_info,
                        'cache_info': cache_info,
                        'original_video_path': video_state.get("original_video_path"),
                        'processed_frame_count': frame_idx - start_frame
                    }
                    tasks.append(task)
            
            # 并行处理
            results = self._process_frames_parallel(tasks, images_dir, labels_dir)
            
            # 统计结果
            successful_frames = len([r for r in results if r['success']])
            total_time = time.time() - start_time
            
            logger.info(f"✅ 数据集生成完成！")
            logger.info(f"📊 处理统计: {successful_frames}/{len(tasks)} 帧成功")
            logger.info(f"⏱️ 总耗时: {total_time:.2f}秒")
            logger.info(f"🚀 平均速度: {successful_frames/total_time:.2f} 帧/秒")
            
            operation_log = [("", ""), (f"高效数据集生成完成！处理了 {successful_frames} 帧，耗时 {total_time:.2f}秒", "Normal")]
            
            return output_dir, operation_log
            
        except Exception as e:
            logger.error(f"❌ 高效数据集生成失败: {str(e)}")
            operation_log = [("", ""), (f"数据集生成失败: {str(e)}", "Error")]
            return None, operation_log
    
    def _get_mask_numbers(self, mask_dropdown: List[str], masks: List[np.ndarray]) -> List[int]:
        """获取mask编号"""
        if len(mask_dropdown) == 0:
            # 自动检测所有unique mask values
            unique_masks = []
            for mask in masks:
                if mask is not None:
                    unique_values = np.unique(mask)
                    unique_masks.extend(unique_values[unique_values > 0])
            
            if len(unique_masks) > 0:
                mask_numbers = sorted(list(set(unique_masks)))
                logger.info(f"🔍 自动检测到的mask编号: {mask_numbers}")
            else:
                raise ValueError("未找到追踪对象")
        else:
            # 用户选择的mask
            mask_dropdown.sort()
            mask_numbers = [int(mask_dropdown[i].split("_")[1]) for i in range(len(mask_dropdown))]
            logger.info(f"🎯 用户选择的mask编号: {mask_numbers}")
        
        return mask_numbers
    
    def _get_processing_range(self, video_state: Dict, interactive_state: Dict, total_frames: int) -> Tuple[int, int]:
        """获取处理范围"""
        start_frame = video_state["select_frame_number"]
        if interactive_state.get("track_end_number"):
            end_frame = interactive_state["track_end_number"]
        else:
            end_frame = total_frames
        
        logger.info(f"📹 处理范围: 从第 {start_frame} 帧到第 {end_frame-1} 帧，共 {end_frame - start_frame} 帧")
        return start_frame, end_frame
    
    def _create_output_directories(self, output_dir: str) -> Tuple[str, str]:
        """创建输出目录"""
        os.makedirs("./temp_yolo_datasets", exist_ok=True)
        images_dir = os.path.join(output_dir, "images")
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        logger.info(f"📁 创建输出目录: {output_dir}")
        return images_dir, labels_dir
    
    def _process_frames_parallel(self, tasks: List[Dict], images_dir: str, labels_dir: str) -> List[Dict]:
        """并行处理帧"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self._process_single_frame, task, images_dir, labels_dir): task 
                for task in tasks
            }
            
            # 收集结果
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success'] and result['processed_frame_count'] % 20 == 0:
                        logger.info(f"📊 已处理 {result['processed_frame_count']} 帧")
                        
                except Exception as e:
                    logger.error(f"处理帧 {task['frame_idx']} 失败: {e}")
                    results.append({
                        'frame_idx': task['frame_idx'],
                        'success': False,
                        'error': str(e),
                        'processed_frame_count': task['processed_frame_count']
                    })
        
        return results
    
    def _process_single_frame(self, task: Dict, images_dir: str, labels_dir: str) -> Dict:
        """处理单个帧"""
        try:
            frame_idx = task['frame_idx']
            frame = task['frame']
            mask = task['mask']
            mask_numbers = task['mask_numbers']
            need_coordinate_conversion = task['need_coordinate_conversion']
            transcode_info = task['transcode_info']
            cache_info = task['cache_info']
            original_video_path = task['original_video_path']
            processed_frame_count = task['processed_frame_count']
            
            if frame is None or mask is None:
                return {
                    'frame_idx': frame_idx,
                    'success': False,
                    'error': 'Frame or mask is None',
                    'processed_frame_count': processed_frame_count
                }
            
            # 保存帧图像
            frame_path = self._save_frame_image(
                frame, processed_frame_count, images_dir, 
                need_coordinate_conversion, cache_info, frame_idx, original_video_path
            )
            
            # 生成YOLO标注
            yolo_annotations = self._generate_yolo_annotations(
                frame, mask, mask_numbers, need_coordinate_conversion, transcode_info
            )
            
            # 保存标注文件
            label_path = os.path.join(labels_dir, f"frame_{processed_frame_count:06d}.txt")
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            return {
                'frame_idx': frame_idx,
                'success': True,
                'frame_path': frame_path,
                'label_path': label_path,
                'annotations_count': len(yolo_annotations),
                'processed_frame_count': processed_frame_count
            }
            
        except Exception as e:
            return {
                'frame_idx': task['frame_idx'],
                'success': False,
                'error': str(e),
                'processed_frame_count': task['processed_frame_count']
            }
    
    def _save_frame_image(self, frame: np.ndarray, processed_frame_count: int, 
                         images_dir: str, need_coordinate_conversion: bool, 
                         cache_info: Dict, frame_idx: int, original_video_path: str) -> str:
        """保存帧图像"""
        if need_coordinate_conversion and cache_info:
            try:
                # 从缓存高效获取原始高分辨率帧
                original_frame_rgb = self.video_processor.get_original_frame_efficient(frame_idx, cache_info)
                frame_pil = PIL.Image.fromarray(original_frame_rgb)
            except Exception as e:
                logger.warning(f"缓存读取失败，回退到直接读取: {e}")
                # 回退到原来的方法
                original_cap = cv2.VideoCapture(original_video_path)
                original_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, original_frame = original_cap.read()
                original_cap.release()
                
                if ret:
                    original_frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                    frame_pil = PIL.Image.fromarray(original_frame_rgb)
                else:
                    frame_pil = PIL.Image.fromarray(frame)
        else:
            frame_pil = PIL.Image.fromarray(frame)
        
        frame_path = os.path.join(images_dir, f"frame_{processed_frame_count:06d}.jpg")
        frame_pil.save(frame_path)
        
        return frame_path
    
    def _generate_yolo_annotations(self, frame: np.ndarray, mask: np.ndarray, 
                                 mask_numbers: List[int], need_coordinate_conversion: bool, 
                                 transcode_info: Dict) -> List[str]:
        """生成YOLO标注"""
        yolo_annotations = []
        
        for mask_num in mask_numbers:
            if mask_num == 0:  # Skip background
                continue
                
            # Create binary mask for this object
            object_mask = (mask == mask_num).astype(np.uint8)
            
            if np.sum(object_mask) > 0:  # If object exists in this frame
                # Find bounding box
                rows = np.any(object_mask, axis=1)
                cols = np.any(object_mask, axis=0)
                
                if len(np.where(rows)[0]) > 0 and len(np.where(cols)[0]) > 0:
                    y1, y2 = np.where(rows)[0][[0, -1]]
                    x1, x2 = np.where(cols)[0][[0, -1]]
                    
                    # Convert to YOLO format (normalized coordinates)
                    height, width = frame.shape[:2]
                    x_center = (x1 + x2) / 2 / width
                    y_center = (y1 + y2) / 2 / height
                    w = (x2 - x1) / width
                    h = (y2 - y1) / height
                    
                    # 如果需要坐标转换，将坐标从处理视频转换到原始视频
                    if need_coordinate_conversion and transcode_info:
                        bbox = [x_center, y_center, w, h]
                        converted_bbox = self.video_processor.convert_bbox_coordinates(
                            bbox, transcode_info, direction="resized_to_original"
                        )
                        x_center, y_center, w, h = converted_bbox
                    
                    # Use mask number as class ID
                    yolo_annotations.append(f"{mask_num} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        
        return yolo_annotations

# 全局高效数据集生成器实例
efficient_generator = EfficientDatasetGenerator()

def get_efficient_generator() -> EfficientDatasetGenerator:
    """获取全局高效数据集生成器实例"""
    return efficient_generator

