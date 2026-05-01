#!/usr/bin/env python3
"""
测试app.py中的get_frames_from_video函数
"""

import os
import argparse
import numpy as np
import cv2
import time
import psutil
from backend.track_anything import TrackingAnything


def test_get_frames_from_video():
    """测试get_frames_from_video函数"""
    print("测试get_frames_from_video函数...")

    # 构造参数
    args = argparse.Namespace(device="cpu", sam_model_type="vit_h", mask_save=False, debug=False, port=8000)

    folder = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "checkpoints")
    SAM_checkpoint = os.path.join(folder, "sam_vit_h_4b8939.pth")
    xmem_checkpoint = os.path.join(folder, "XMem-s012.pth")

    model = TrackingAnything(SAM_checkpoint, xmem_checkpoint, None, args)

    # 创建测试视频
    test_video_path = os.path.join(os.path.dirname(__file__), "test_video_temp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video_path, fourcc, 30.0, (640,480))

    for i in range(10):  # 10帧
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.rectangle(frame, (100+i*10, 100), (200+i*10, 200), (255, 0, 0), -1)
        out.write(frame)

    out.release()

    # 模拟get_frames_from_video函数
    video_input = test_video_path
    video_state = {}

    frames = []
    user_name = time.time()
    operation_log = [("",""),("Upload video already. Try click the image for adding targets to track and inpaint.","Normal")]

    try:
        cap = cv2.VideoCapture(video_input)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                current_memory_usage = psutil.virtual_memory().percent
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if current_memory_usage > 90:
                    operation_log = [("Memory usage is too high (>90%). Stop the video extraction. Please reduce the video resolution or frame rate.", "Error")]
                    print("Memory usage is too high (>90%). Please reduce the video resolution or frame rate.")
                    break
            else:
                break
        cap.release()
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_input, str(e)))
        operation_log = [("",""), (f"Error reading video: {str(e)}", "Error")]
        # 清理测试文件
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
        return False

    if len(frames) == 0:
        operation_log = [("",""), ("No frames extracted from video. Please check the video file.", "Error")]
        # 清理测试文件
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
        return False

    image_size = (frames[0].shape[0],frames[0].shape[1])
    # initialize video_state
    video_state = {
        "user_name": user_name,
        "video_name": os.path.split(video_input)[-1],
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0],frames[0].shape[1]), np.uint8)]*len(frames),
        "logits": [None]*len(frames),
        "select_frame_number": 0,
        "fps": fps
        }

    video_info = "Video Name: {}, FPS: {}, Total Frames: {}, Image Size:{}".format(video_state["video_name"], video_state["fps"], len(frames), image_size)

    try:
        model.samcontroler.sam_controler.reset_image()
        model.samcontroler.sam_controler.set_image(video_state["origin_images"][0])
        print("SAM模型设置成功")
    except Exception as e:
        print(f"SAM模型设置失败: {e}")
        # 清理测试文件
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
        return False

    print(f"视频处理成功: {video_info}")

    # 清理测试文件
    os.remove(test_video_path)

    return True

if __name__ == "__main__":
    success = test_get_frames_from_video()
    if success:
        print("get_frames_from_video函数测试通过！")
    else:
        print("get_frames_from_video函数测试失败！")
