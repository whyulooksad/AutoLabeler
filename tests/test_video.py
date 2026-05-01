#!/usr/bin/env python3
"""
测试视频处理功能
"""

import os
import numpy as np
import cv2


def test_video_processing():
    """测试视频处理功能"""
    print("测试视频处理功能...")

    # 创建一个简单的测试视频
    test_video_path = os.path.join(os.path.dirname(__file__), "test_video_temp.mp4")

    # 创建测试视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video_path, fourcc, 30.0, (640,480))

    for i in range(30):  # 30帧
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # 添加一个移动的矩形
        cv2.rectangle(frame, (100+i*10, 100), (200+i*10, 200), (255, 0, 0), -1)
        out.write(frame)

    out.release()
    print(f"创建测试视频: {test_video_path}")

    # 测试视频读取
    try:
        cap = cv2.VideoCapture(test_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                break

        cap.release()

        print(f"成功读取视频: {len(frames)} 帧, FPS: {fps}")

        # 清理测试文件
        os.remove(test_video_path)

        return True

    except Exception as e:
        print(f"视频处理测试失败: {e}")
        # 清理测试文件
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
        return False

if __name__ == "__main__":
    success = test_video_processing()
    if success:
        print("视频处理测试通过！")
    else:
        print("视频处理测试失败！")
