#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试抽帧数据集生成器
"""

import os
import numpy as np
import logging
import time
from backend.sampled_dataset_generator import SampledDatasetGenerator

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TEST_SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "test_sample")


def create_test_data(num_frames: int = 100, frame_size: tuple = (1280, 720)) -> dict:
    """创建测试数据"""
    logger.info(f"创建测试数据: {num_frames} 帧，尺寸: {frame_size}")

    # 创建模拟视频状态
    frames = []
    masks = []

    for i in range(num_frames):
        # 创建模拟帧
        frame = np.random.randint(0, 255, (*frame_size, 3), dtype=np.uint8)
        frames.append(frame)

        # 创建模拟mask（包含2个对象）
        mask = np.zeros(frame_size, dtype=np.uint8)
        # 对象1
        mask[200:400, 300:500] = 1
        # 对象2
        mask[600:800, 700:900] = 2
        masks.append(mask)

    video_state = {
        "origin_images": frames,
        "masks": masks,
        "select_frame_number": 0
    }

    interactive_state = {
        "track_end_number": num_frames
    }

    mask_dropdown = ["mask_1", "mask_2"]

    return video_state, interactive_state, mask_dropdown

def test_sampled_generator():
    """测试抽帧数据集生成器"""
    logger.info("测试抽帧数据集生成器")
    logger.info("=" * 60)

    # 创建测试数据
    video_state, interactive_state, mask_dropdown = create_test_data(100, (1280, 720))

    # 测试不同的抽帧间隔
    test_cases = [
        {"sample_interval": 4, "name": "每4帧抽1帧"},
        {"sample_interval": 8, "name": "每8帧抽1帧"},
        {"sample_interval": 16, "name": "每16帧抽1帧"},
    ]

    for test_case in test_cases:
        logger.info(f"\n测试 {test_case['name']}")
        logger.info("-" * 40)

        # 创建生成器
        generator = SampledDatasetGenerator(
            max_workers=4,
            sample_interval=test_case["sample_interval"]
        )

        # 生成数据集
        start_time = time.time()
        output_dir, operation_log = generator.generate_yolo_dataset_sampled(
            video_state, interactive_state, mask_dropdown
        )
        total_time = time.time() - start_time

        if output_dir:
            logger.info(f"数据集生成成功: {output_dir}")
            logger.info(f"耗时: {total_time:.2f}秒")

            # 检查生成的文件
            images_dir = os.path.join(output_dir, "images")
            labels_dir = os.path.join(output_dir, "labels")

            if os.path.exists(images_dir):
                image_count = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
                logger.info(f"生成图像数量: {image_count}")

            if os.path.exists(labels_dir):
                label_count = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
                logger.info(f"生成标注数量: {label_count}")

            # 检查配置文件
            config_path = os.path.join(output_dir, "dataset.yaml")
            if os.path.exists(config_path):
                logger.info(f"配置文件生成: {config_path}")
        else:
            logger.error("数据集生成失败")

def test_performance_comparison():
    """性能对比测试"""
    logger.info("\n" + "=" * 60)
    logger.info("性能对比测试")
    logger.info("=" * 60)

    # 创建测试数据
    video_state, interactive_state, mask_dropdown = create_test_data(200, (1280, 720))

    # 测试不同抽帧间隔的性能
    intervals = [1, 4, 8, 16]
    results = []

    for interval in intervals:
        logger.info(f"\n测试抽帧间隔: {interval}")

        generator = SampledDatasetGenerator(max_workers=4, sample_interval=interval)

        start_time = time.time()
        output_dir, _ = generator.generate_yolo_dataset_sampled(
            video_state, interactive_state, mask_dropdown
        )
        total_time = time.time() - start_time

        if output_dir:
            # 计算实际处理的帧数
            images_dir = os.path.join(output_dir, "images")
            if os.path.exists(images_dir):
                image_count = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])

                result = {
                    "interval": interval,
                    "time": total_time,
                    "frames": image_count,
                    "speed": image_count / total_time if total_time > 0 else 0
                }
                results.append(result)

                logger.info(f"  处理帧数: {image_count}")
                logger.info(f"  耗时: {total_time:.2f}秒")
                logger.info(f"  速度: {result['speed']:.2f} 帧/秒")

    # 输出总结
    logger.info("\n性能总结:")
    for result in results:
        logger.info(f"  间隔{result['interval']}: {result['frames']}帧, {result['time']:.2f}秒, {result['speed']:.2f}帧/秒")

if __name__ == "__main__":
    # 运行基本功能测试
    test_sampled_generator()

    # 运行性能对比测试
    test_performance_comparison()

    logger.info("\n测试完成！")
