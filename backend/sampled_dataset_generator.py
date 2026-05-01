#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽帧数据集生成器
基于处理后的视频B生成数据集，支持抽帧处理
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

class SampledDatasetGenerator:
    """抽帧数据集生成器"""
    
    def __init__(self, max_workers: int = None, sample_interval: int = 8):
        """
        初始化数据集生成器
        
        Args:
            max_workers: 最大工作线程数，默认为CPU核心数
            sample_interval: 抽帧间隔，每N帧抽取1帧
        """
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.sample_interval = sample_interval
        self.video_processor = VideoProcessor(target_width=1280)
        
    def generate_yolo_dataset_sampled(self, video_state: Dict, interactive_state: Dict, mask_dropdown: List[str], output_dir: str = None) -> Tuple[str, List]:
        """基于视频B生成抽帧YOLO数据集"""
        operation_log = [("", ""), ("开始生成抽帧YOLO数据集...", "Normal")]
        
        try:
            start_time = time.time()
            
            # 验证输入数据
            if not video_state or not video_state.get("origin_images"):
                raise ValueError("没有可用的视频数据")
            
            frames = video_state["origin_images"]
            masks = video_state["masks"]
            class_names = interactive_state.get("class_names") or video_state.get("class_names")
            mask_class_ids = interactive_state.get("mask_class_ids") or video_state.get("mask_class_ids") or {}
            
            mask_numbers = self._get_mask_numbers(mask_dropdown, masks)
            if not class_names:
                class_names = [f"object_{max(0, mask_num - 1)}" for mask_num in mask_numbers] or ["object"]
            class_id_by_mask_number = self._get_class_id_by_mask_number(mask_numbers, mask_class_ids)
            
            # 确定处理范围
            start_frame, end_frame = self._get_processing_range(video_state, interactive_state, len(frames))
            
            # 获取抽帧索引
            sampled_indices = self._get_sampled_indices(start_frame, end_frame)
            
            logger.info(f"📹 处理范围: 从第 {start_frame} 帧到第 {end_frame-1} 帧")
            logger.info(f"🎯 抽帧设置: 每 {self.sample_interval} 帧抽取1帧")
            logger.info(f"📊 抽帧结果: 从 {len(range(start_frame, end_frame))} 帧中抽取 {len(sampled_indices)} 帧")
            
            # 创建输出目录
            if output_dir is None:
                output_dir = f"./temp_yolo_datasets/yolo_dataset_sampled_{int(time.time())}"
            
            images_dir, labels_dir = self._create_output_directories(output_dir)
            
            # 批量处理抽帧
            logger.info(f"🚀 开始批量处理 {len(sampled_indices)} 帧，使用 {self.max_workers} 个线程")
            
            # 准备批量处理任务
            tasks = []
            for i, frame_idx in enumerate(sampled_indices):
                if frame_idx < len(frames) and frame_idx < len(masks):
                    task = {
                        'frame_idx': frame_idx,
                        'frame': frames[frame_idx],
                        'mask': masks[frame_idx],
                        'mask_numbers': mask_numbers,
                        'class_id_by_mask_number': class_id_by_mask_number,
                        'processed_frame_count': i
                    }
                    tasks.append(task)
            
            # 并行处理
            results = self._process_frames_parallel(tasks, images_dir, labels_dir)
            
            # 统计结果
            successful_frames = len([r for r in results if r['success']])
            total_time = time.time() - start_time
            
            logger.info(f"✅ 抽帧数据集生成完成！")
            logger.info(f"📊 处理统计: {successful_frames}/{len(tasks)} 帧成功")
            logger.info(f"⏱️ 总耗时: {total_time:.2f}秒")
            logger.info(f"🚀 平均速度: {successful_frames/total_time:.2f} 帧/秒")
            
            # 生成数据集配置文件
            self._generate_dataset_config(output_dir, class_names)
            
            operation_log = [("", ""), (f"抽帧数据集生成完成！处理了 {successful_frames} 帧，耗时 {total_time:.2f}秒", "Normal")]
            
            return output_dir, operation_log
            
        except Exception as e:
            logger.error(f"❌ 抽帧数据集生成失败: {str(e)}")
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
        
        return start_frame, end_frame
    
    def _get_sampled_indices(self, start_frame: int, end_frame: int) -> List[int]:
        """获取抽帧索引"""
        frame_range = list(range(start_frame, end_frame))
        sampled_indices = frame_range[::self.sample_interval]
        return sampled_indices
    
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
                    
                    if result['success'] and result['processed_frame_count'] % 10 == 0:
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
            class_id_by_mask_number = task['class_id_by_mask_number']
            processed_frame_count = task['processed_frame_count']
            
            if frame is None or mask is None:
                return {
                    'frame_idx': frame_idx,
                    'success': False,
                    'error': 'Frame or mask is None',
                    'processed_frame_count': processed_frame_count
                }
            
            # 保存帧图像
            frame_path = self._save_frame_image(frame, processed_frame_count, images_dir)
            
            # 生成YOLO标注
            yolo_annotations = self._generate_yolo_annotations(frame, mask, mask_numbers, class_id_by_mask_number)
            
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
    
    def _save_frame_image(self, frame: np.ndarray, processed_frame_count: int, images_dir: str) -> str:
        """保存帧图像"""
        frame_pil = PIL.Image.fromarray(frame)
        frame_path = os.path.join(images_dir, f"frame_{processed_frame_count:06d}.jpg")
        frame_pil.save(frame_path)
        
        return frame_path
    
    def _generate_yolo_annotations(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        mask_numbers: List[int],
        class_id_by_mask_number: Dict[int, int],
    ) -> List[str]:
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
                    
                    class_id = class_id_by_mask_number.get(mask_num, max(0, mask_num - 1))
                    yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        
        return yolo_annotations
    
    def _get_class_id_by_mask_number(self, mask_numbers: List[int], mask_class_ids: Dict[str, int]) -> Dict[int, int]:
        class_id_by_mask_number = {}
        for mask_num in mask_numbers:
            class_id_by_mask_number[mask_num] = int(mask_class_ids.get(f"mask_{mask_num:03d}", max(0, mask_num - 1)))
        return class_id_by_mask_number
    
    def _generate_dataset_config(self, output_dir: str, class_names: List[str]):
        """生成数据集配置文件"""
        try:
            names = class_names or ["object"]
            names_str = ", ".join([f"'{name}'" for name in names])
            
            # 创建dataset.yaml文件
            dataset_config = f"""# YOLO Dataset Configuration
# Generated by Track-Anything Sampled Dataset Generator

# Dataset paths
path: {os.path.abspath(output_dir)}
train: images
val: images

# Number of classes
nc: {len(names)}

# Class names
names: [{names_str}]

# Sampling information
sample_interval: {self.sample_interval}
target_width: {self.video_processor.target_width}

# Generation info
generated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}
generator: Track-Anything Sampled Dataset Generator
"""
            
            config_path = os.path.join(output_dir, "dataset.yaml")
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(dataset_config)
            
            logger.info(f"📝 生成数据集配置文件: {config_path}")
            
        except Exception as e:
            logger.error(f"生成数据集配置文件失败: {e}")

# 全局抽帧数据集生成器实例
sampled_generator = SampledDatasetGenerator()

def get_sampled_generator() -> SampledDatasetGenerator:
    """获取全局抽帧数据集生成器实例"""
    return sampled_generator
