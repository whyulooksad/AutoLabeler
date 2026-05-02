# AutoLabeler

## 一、项目介绍

**AutoLabeler** 是一个面向视频目标自动标注的工具，用于从视频中快速生成 YOLO 格式的目标检测数据集。

本项目基于 [Track-Anything](https://github.com/gaomingqi/Track-Anything) 改造而来。**Track-Anything** 定位为交互式视频追踪与修复工具，我保留了其核心的 SAM 点击分割 + XMem 视频追踪能力，并围绕 **标注效率** 做了大量针对性改造——包括 YOLO 数据集导出、帧采样、视频预处理、坐标回映、性能优化等。你只需要在起始帧上通过正/负点提示得到目标 mask，系统会把目标传播到后续帧，并把追踪结果转换成可直接用于 YOLO 训练的数据集。

## 二、环境配置

建议使用 Conda 创建 Python 3.10 环境。当前验证过的环境是：

```
Python 3.10.20
PyTorch 2.7.1+cu118
CUDA 11.8
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

创建环境：

```powershell
conda create -n autolabeler python=3.10 -y
conda activate autolabeler
```

安装 PyTorch CUDA 11.8 版本：

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

安装项目依赖：

```powershell
pip install -r requirements.txt
```

验证 PyTorch 和 CUDA：

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

如果输出里 `torch.cuda.is_available()` 是 `True`，说明 GPU 环境可用。

## 三、模型权重

模型权重放在`checkpoints/`下

主要需要：

| 模型                   | 用途                                    |
| ---------------------- | --------------------------------------- |
| SAM (Segment Anything) | 点击分割——用户在帧上点击，SAM 生成 mask |
| XMem                   | 视频追踪——将 mask 逐帧传播到后续帧      |

如果文件不存在，后端会尝试自动下载。由于权重文件较大，建议提前放好，避免启动或首次上传视频时等待过久。

## 四、使用方法

启动后端：

```powershell
conda activate autolabeler
cd /d E:\Work\AutoLabeler
python app.py
```

后端默认地址：

```text
http://127.0.0.1:8000
```

启动前端：

```powershell
cd /d E:\Work\AutoLabeler\frontend
npm install
npm run dev -- --host 0.0.0.0
```

前端默认地址：

```text
http://127.0.0.1:5173
```

**基本工作流：**

1. 上传视频。

2. 选择起始帧和结束帧。

3. 选择视频追踪时的分辨率：original / 720 / 480 / 360（主要影响跟踪计算精度 + 显存占用 + 速度）

4. 使用 Positive / Negative 点击目标区域，生成 SAM 分割结果。

5. 确认 mask 后点击 Add 保存目标。

6. 选择要追踪的 mask。

   ![9f7673bf-3e61-4f03-b62a-c0367eee6a14](assets\0.png)

7. 点击 Track 生成追踪结果视频。

   ![test-sample1](assets\1.gif)

8. 点击 YOLO 导出 YOLO 格式数据集压缩包。

   ![image-20260502023949292](assets\2.png)

![image-20260502024018763](assets\3.png)

## 五、代码结构

```
AutoLabeler/
├── app.py                            # 应用入口，启动 FastAPI 服务
├── requirements.txt                  # Python 依赖
├── backend/                          # 后端核心代码
│   ├── server.py                     # FastAPI REST API、会话管理
│   ├── track_anything.py             # SAM + XMem 门面类
│   ├── video_processor.py            # 视频转码、坐标映射、帧采样
│   ├── sampled_dataset_generator.py  # 抽帧式 YOLO 数据集生成器
│   ├── efficient_dataset_generator.py# 全帧式 YOLO 数据集生成器（含坐标回映）
│   ├── demo.py                       # metaseg 独立演示脚本
│   ├── tools/                        # SAM 封装与可视化
│   │   ├── base_segmenter.py         #   SAM 模型加载与 predict
│   │   ├── interact_tools.py         #   点击交互控制器 SamControler
│   │   ├── painter.py                #   mask/点 可视化绘制
│   │   └── mask_painter.py           #   mask 绘制（距离变换模式）
│   ├── tracker/                      # XMem 视频追踪
│   │   ├── base_tracker.py           #   追踪接口，串联 InferenceCore
│   │   ├── config/config.yaml        #   XMem 推理配置
│   │   ├── inference/                #   推理核心与记忆管理
│   │   │   ├── inference_core.py     #     逐帧推理循环
│   │   │   ├── memory_manager.py     #     工作记忆 + 长期记忆管理
│   │   │   └── kv_memory_store.py    #     KV 记忆存储
│   │   ├── model/                    #   XMem 网络结构
│   │   │   ├── network.py            #     XMem 主模型（encode_key/encode_value/segment）
│   │   │   ├── modules.py            #     编码器、解码器、KeyProjection 等组件
│   │   │   ├── resnet.py             #     支持额外输入通道的 ResNet
│   │   │   ├── group_modules.py      #     多目标分组卷积/上采样
│   │   │   ├── cbam.py               #     通道+空间注意力模块
│   │   │   ├── aggregate.py          #     多目标概率软聚合
│   │   │   ├── memory_util.py        #     相似度计算、softmax、readout
│   │   │   ├── losses.py             #     训练损失（Dice + BootstrappedCE）
│   │   │   └── trainer.py            #     分布式训练封装
│   │   └── util/                     #   工具函数
│   │       ├── mask_mapper.py        #     索引 mask → one-hot 转换与标签重映射
│   │       ├── range_transform.py    #     ImageNet 归一化
│   │       └── tensor_util.py        #     padding/unpadding、IoU
│   ├── inpainter/                    # E2FGVI 视频修复（已禁用，保留代码）
│   │   ├── base_inpainter.py         #   修复接口，加载 E2FGVI-HQ
│   │   ├── config/config.yaml        #   修复推理配置
│   │   ├── model/                    #   E2FGVI 网络结构
│   │   │   ├── e2fgvi.py             #     标准版生成器 + 判别器
│   │   │   ├── e2fgvi_hq.py          #     HQ 版（支持可变分辨率）
│   │   │   └── modules/              #     子模块
│   │   │       ├── feat_prop.py      #       双向特征传播
│   │   │       ├── flow_comp.py      #       SPyNet 光流估计 + flow_warp
│   │   │       ├── tfocal_transformer.py      # 标准版时序焦点 Transformer
│   │   │       ├── tfocal_transformer_hq.py   # HQ 版时序焦点 Transformer
│   │   │       └── spectral_norm.py          # 谱归一化
│   │   └── util/
│   │       └── tensor_util.py        #   帧/mask 缩放工具
│   └── stub_mmcv/                    # mmcv 空桩模块，防止导入报错
│       └── mmcv.py
├── frontend/                         # React 前端
│   └── src/
│       ├── main.tsx                  #   前端应用主体
│       └── styles.css                #   样式
├── scripts/                          # 工具脚本
│   ├── check_gpu.py                  #   GPU/CUDA 环境检测
│   ├── quick_test_coordinates.py     #   坐标转换验证
│   ├── performance_comparison.py     #   生成器性能对比
│   ├── backup_project.py             #   项目源码备份
│   ├── backup_project_full.py        #   完整备份（含权重）
│   ├── quick_backup.py               #   交互式备份菜单
│   └── backup_manager.py             #   备份管理（列表/删除/恢复）
├── tests/                            # pytest 测试
│   ├── test_video.py                 #   视频处理测试
│   ├── test_video_processor.py       #   VideoProcessor 测试
│   ├── test_sampled_generator.py     #   采样生成器测试
│   ├── test_track.py                 #   SAM + XMem 集成测试
│   └── test_app_function.py         #   上传/初始化流程测试
├── checkpoints/                      # SAM 和 XMem 模型权重
├── result/                           # 追踪结果输出
├── temp_uploads/                     # 上传视频临时目录
├── temp_videos/                      # 转码后的视频缓存
├── temp_yolo_datasets/               # YOLO 数据集导出目录
└── test_sample/                      # 示例视频
```


## 六、 LICENSE

本项目基于 [MIT License](LICENSE) 开源。
