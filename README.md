# Nutri-Pal

基于 YOLOv8 目标检测与机器学习，集成 LabelImg 数据标注、OpenCV 视觉处理和简易 GUI 窗口的营养健康伙伴。

## 项目简介

本项目旨在通过目标检测与视觉处理技术，识别食品或食材、估算数量/大小并辅助营养分析。项目主要使用 Python 实现检测与业务逻辑，同时包含用于性能优化/硬件接口的 C++ / CUDA 代码。

主要功能（示例）：
- 使用 YOLOv8 模型进行目标检测
- 使用 OpenCV 进行图像预处理与可视化
- 集成 LabelImg 的标注流程，便于生成训练数据
- 提供简易 GUI 窗口用于演示/调试

## 目录（示例）

- dataset/         —— 数据集、标注文件（VOC / YOLO 格式）
- models/          —— 训练好的权重与模型配置
- src/             —— Python 源码（检测、后处理、GUI）
- cpp/             —— C++ 源码（可选的性能模块）
- tools/           —— 标注、转换、评估等工具脚本
- requirements.txt —— Python 依赖（如果存在）

（实际目录以仓库内容为准，请根据项目结构调整）

## 依赖与安装

建议在虚拟环境中安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
.venv\Scripts\activate     # Windows (PowerShell)

# 常见依赖（若仓库提供 requirements.txt，请使用它）
pip install -U pip
pip install -r requirements.txt  # 如果存在
# 或者单独安装常用包
pip install ultralytics opencv-python numpy
```

C++ 模块（如果存在）可使用 CMake 构建：

```bash
mkdir build && cd build
cmake ..
make -j
```

如果需要 GPU/CUDA 支持，请确保已安装相应版本的 CUDA 与驱动，并用匹配的依赖（如 torch）安装支持 GPU 的包。

## 使用示例

以下为常见流程示例，请根据仓库中实际脚本修改命令：

1. 数据标注（使用 LabelImg）

- 启动 LabelImg，标注并保存为 YOLO/VOC 格式，然后将标注文件放入 dataset/ 下。

2. 训练模型（示例）

```bash
# 使用 ultralytics/YOLOv8 的训练示例（修改为项目中的配置文件）
python train.py --data dataset/data.yaml --cfg models/yolov8n.yaml --epochs 100 --img 640
```

3. 推理/检测（示例）

```bash
python detect.py --weights models/best.pt --source examples/images --save-txt
```

4. 运行 GUI（如果有）

```bash
python main.py
```

## 数据与模型格式

- 建议使用 YOLO 格式（.txt + 图像）或 VOC 格式（.xml）保存标注
- 模型权重推荐保存为 .pt（PyTorch / ultralytics）

## 贡献

欢迎提交 issue 与 pull request：

- 提交 bug 请附带复现步骤与相关日志
- 若要新增功能，请先在 issue 中讨论设计方案

<img width="2607" height="3445" alt="流程图" src="https://github.com/user-attachments/assets/0bd635bc-f9fb-46cd-a5a8-e9dd20256eaf" />
<img width="8833" height="1136" alt="流程图 (2)" src="https://github.com/user-attachments/assets/96fc63e6-d7b7-4991-9e80-17681b5ea7b9" />
<img width="1600" height="1600" alt="labels" src="https://github.com/user-attachments/assets/e81f09a0-5736-436f-9d0e-acc34492a8d5" />
<img width="1920" height="1920" alt="val_batch0_labels" src="https://github.com/user-attachments/assets/725ebbab-b9ff-4442-8672-627cc726da3c" />
<img width="1920" height="1920" alt="train_batch82" src="https://github.com/user-attachments/assets/2fefc3c2-e941-43c0-997f-96aeb2ee3338" />


