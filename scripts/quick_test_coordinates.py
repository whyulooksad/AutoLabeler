#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试坐标转换功能
"""

import numpy as np
import _path_setup
from backend.video_processor import VideoProcessor

def test_coordinate_conversion():
    """测试坐标转换功能"""
    print("🧪 测试坐标转换功能")
    print("=" * 50)
    
    # 创建视频处理器
    processor = VideoProcessor(target_width=640)
    
    # 模拟转码信息 (1920x1080 -> 640x360)
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
    
    # 测试坐标点
    test_coordinates = [
        [0.5, 0.5, 0.2, 0.2],  # 中心点
        [0.0, 0.0, 0.1, 0.1],  # 左上角
        [1.0, 1.0, 0.1, 0.1],  # 右下角
    ]
    
    for i, coord in enumerate(test_coordinates):
        print(f"测试坐标 {i+1}: {coord}")
        
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
        
        print(f"  转换后: {converted}")
        print(f"  反向转换: {back_converted}")
        print(f"  最大误差: {max_diff:.6f}")
        
        if max_diff < 1e-5:
            print(f"  ✅ 坐标 {i+1} 转换准确")
        else:
            print(f"  ⚠️ 坐标 {i+1} 转换有误差")
        print()

if __name__ == "__main__":
    test_coordinate_conversion()

