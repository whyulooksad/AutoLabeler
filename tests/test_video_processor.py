#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频处理器测试脚本
测试视频转码和坐标转换功能
"""

import os
import cv2
import numpy as np
import logging
from backend.video_processor import VideoProcessor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TEST_SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "test_sample")


def test_video_processor():
    """测试视频处理器功能"""

    # 使用test_sample中的视频，如果有的话
    test_videos = [f for f in os.listdir(TEST_SAMPLE_DIR) if f.endswith(".mp4")] if os.path.isdir(TEST_SAMPLE_DIR) else []

    if test_videos:
        test_video_path = os.path.join(TEST_SAMPLE_DIR, test_videos[0])
        logger.info(f"使用test_sample中的视频: {test_video_path}")
        use_temp = False
    else:
        # 创建测试视频
        test_video_path = create_test_video()
        use_temp = True

    try:
        # 初始化视频处理器
        processor = VideoProcessor(target_width=640)

        # 测试1: 获取视频信息
        logger.info("=" * 50)
        logger.info("测试1: 获取视频信息")
        video_info = processor.get_video_info(test_video_path)
        logger.info(f"视频信息: {video_info}")

        # 测试2: 计算缩放比例
        logger.info("=" * 50)
        logger.info("测试2: 计算缩放比例")
        scale_ratio, target_width, target_height = processor.calculate_resize_ratio(
            video_info["original_width"],
            video_info["original_height"]
        )
        logger.info(f"缩放比例: {scale_ratio}")
        logger.info(f"目标尺寸: {target_width}x{target_height}")

        # 测试3: 视频转码
        logger.info("=" * 50)
        logger.info("测试3: 视频转码")
        resized_video_path, transcode_info = processor.transcode_video(test_video_path)
        logger.info(f"转码后视频路径: {resized_video_path}")
        logger.info(f"转码信息: {transcode_info}")

        # 测试4: 坐标转换
        logger.info("=" * 50)
        logger.info("测试4: 坐标转换")

        # 测试边界框坐标转换
        test_bbox = [0.5, 0.5, 0.2, 0.3]  # [x_center, y_center, width, height]
        logger.info(f"原始边界框: {test_bbox}")

        # 从转码视频坐标转换到原始视频坐标
        converted_bbox = processor.convert_bbox_coordinates(
            test_bbox, transcode_info, direction="resized_to_original"
        )
        logger.info(f"转换后边界框: {converted_bbox}")

        # 从原始视频坐标转换到转码视频坐标
        back_converted_bbox = processor.convert_bbox_coordinates(
            converted_bbox, transcode_info, direction="original_to_resized"
        )
        logger.info(f"反向转换边界框: {back_converted_bbox}")

        # 测试mask坐标转换
        logger.info("=" * 50)
        logger.info("测试5: Mask坐标转换")

        # 创建测试mask
        test_mask = np.zeros((480, 640), dtype=np.uint8)
        test_mask[100:200, 150:250] = 1  # 创建一个矩形区域

        logger.info(f"原始mask尺寸: {test_mask.shape}")

        # 从转码视频mask转换到原始视频mask
        converted_mask = processor.convert_mask_coordinates(
            test_mask, transcode_info, direction="resized_to_original"
        )
        logger.info(f"转换后mask尺寸: {converted_mask.shape}")

        # 从原始视频mask转换到转码视频mask
        back_converted_mask = processor.convert_mask_coordinates(
            converted_mask, transcode_info, direction="original_to_resized"
        )
        logger.info(f"反向转换mask尺寸: {back_converted_mask.shape}")

        logger.info("=" * 50)
        logger.info("所有测试通过！")

        # 清理
        if use_temp:
            cleanup_test_files(test_video_path, resized_video_path)

    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

def create_test_video():
    """创建测试视频"""
    test_video_path = os.path.join(os.path.dirname(__file__), "test_video_temp.mp4")

    # 创建高分辨率测试视频 (1920x1080)
    width, height = 1920, 1080
    fps = 30
    duration = 3  # 3秒

    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video_path, fourcc, fps, (width, height))

    logger.info(f"创建测试视频: {width}x{height}, {fps}fps, {duration}秒")

    for frame_num in range(fps * duration):
        # 创建渐变背景
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 添加渐变效果
        for y in range(height):
            for x in range(width):
                frame[y, x] = [
                    int(255 * x / width),  # 红色渐变
                    int(255 * y / height), # 绿色渐变
                    int(255 * frame_num / (fps * duration))  # 蓝色随时间变化
                ]

        # 添加移动的圆形
        center_x = int(width / 2 + 200 * np.sin(frame_num * 0.1))
        center_y = int(height / 2 + 100 * np.cos(frame_num * 0.1))
        cv2.circle(frame, (center_x, center_y), 50, (255, 255, 255), -1)

        out.write(frame)

    out.release()
    logger.info(f"测试视频创建完成: {test_video_path}")

    return test_video_path

def cleanup_test_files(*file_paths):
    """清理测试文件"""
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"已删除测试文件: {file_path}")

def test_coordinate_accuracy():
    """测试坐标转换的准确性"""
    logger.info("=" * 50)
    logger.info("测试坐标转换准确性")

    # 创建视频处理器
    processor = VideoProcessor(target_width=640)

    # 模拟转码信息
    transcode_info = {
        "original_info": {
            "original_width": 1920,
            "original_height": 1080
        },
        "resized_info": {
            "original_width": 640,
            "original_height": 360
        },
        "scale_ratio": 640 / 1920
    }

    # 测试多个坐标点
    test_coordinates = [
        [0.0, 0.0, 0.1, 0.1],  # 左上角
        [0.5, 0.5, 0.2, 0.2],  # 中心
        [1.0, 1.0, 0.1, 0.1],  # 右下角
        [0.25, 0.75, 0.3, 0.2], # 左下角
        [0.75, 0.25, 0.2, 0.3]  # 右上角
    ]

    for i, coord in enumerate(test_coordinates):
        logger.info(f"测试坐标 {i+1}: {coord}")

        # 正向转换
        converted = processor.convert_bbox_coordinates(
            coord, transcode_info, direction="resized_to_original"
        )

        # 反向转换
        back_converted = processor.convert_bbox_coordinates(
            converted, transcode_info, direction="original_to_resized"
        )

        # 检查转换精度
        diff = np.array(coord) - np.array(back_converted)
        max_diff = np.max(np.abs(diff))

        logger.info(f"  转换后: {converted}")
        logger.info(f"  反向转换: {back_converted}")
        logger.info(f"  最大误差: {max_diff:.6f}")

        if max_diff < 1e-5:
            logger.info(f"  坐标 {i+1} 转换准确")
        else:
            logger.warning(f"  坐标 {i+1} 转换有误差")

if __name__ == "__main__":
    logger.info("开始测试视频处理器...")

    # 测试基本功能
    test_video_processor()

    # 测试坐标转换准确性
    test_coordinate_accuracy()

    logger.info("测试完成！")
