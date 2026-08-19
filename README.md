# Nutri-Pal —— AI 营养健康伙伴

> 基于 YOLOv8 目标检测 + 自建食物数据集 + OpenCV 视觉处理 + 营养推荐逻辑的全栈饮食健康管理系统

![Python](https://img.shields.io/badge/python-3.10+-blue) ![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-red) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)

---

## 📌 项目背景

日常饮食记录效率低、营养分析碎片化是常见痛点。本项目独立打造全栈饮食健康管理系统，涵盖前端 GUI、后端业务逻辑与 AI 算法实现，完整跑通"**饮食记录录入 → 多维度营养自动计算 → 健康数据可视化 → 个性化膳食建议**"的端到端功能链路。

---

## 🎯 核心功能

- **🍎 YOLOv8 食物识别**：基于自建数据集训练，30 类常见食物目标检测
- **📊 营养成分自动计算**：识别后调用食物-营养数据库，自动计算热量/蛋白质/脂肪/碳水
- **📈 健康数据可视化**：周/月饮食趋势、营养均衡度评分
- **💡 个性化膳食建议**：基于个人健康数据生成推荐
- **🖼️ 简易 GUI 窗口**：基于 OpenCV + Tkinter 的本地交互界面
- **🏷️ LabelImg 标注流程**：完整支持 YOLO/VOC 格式的数据标注工作流

---

## 🛠️ 技术栈

| 组件 | 技术 | 用途 |
|---|---|---|
| 目标检测 | YOLOv8 (ultralytics) | 食物识别 |
| 视觉处理 | OpenCV | 图像预处理与可视化 |
| 数据标注 | LabelImg | 训练数据生成 |
| 深度学习 | PyTorch | 模型训练推理 |
| 训练框架 | PaddleDetection (辅助) | HRNet 分割对比实验 |
| GUI | OpenCV + Tkinter | 本地交互界面 |
| 业务逻辑 | Python | 营养计算、推荐算法 |

---

## 📊 模型与数据

| 指标 | 数值 |
|---|---|
| 模型 | YOLOv8n |
| 数据集规模 | 自建 3000+ 张标注图像 |
| 食物类别 | 30 类 |
| 训练 epoch | 200 |
| mAP@0.5 | 0.87 |
| 推理速度 | ~30 FPS（CPU）/ ~120 FPS（GPU） |

**数据集构建流程**：
1. 网络爬取 + 实拍采集原始食物图像
2. LabelImg 标注（YOLO 格式 + VOC 格式双备份）
3. 数据增强：mosaic / mixup / HSV 调整 / 旋转
4. 划分 train/val/test (8:1:1)

---

## 📁 项目结构

```
nutri-pal/
├── datasets/                  # 自建食物数据集（YOLO 格式）
│   ├── images/
│   └── labels/
├── runs/detect/               # YOLOv8 训练输出
│   ├── train/
│   └── predict/
├── main.py                    # GUI 入口（主应用）
├── main_new.py                # 新版 GUI 入口
├── data.yaml                  # YOLO 数据集配置
├── yolov8n.pt                 # YOLOv8n 预训练权重
├── best.pt                    # 训练后的最终权重
├── hrnet18_ocr64_cocolvis.pdparams  # PaddleDetection 实验
├── PaddleDetection/           # PaddleDetection 训练框架（用于对比实验）
├── requirements.txt
└── README.md
```

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Supu7796/nutri-pal.git
cd nutri-pal
```

### 2. 安装依赖

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install ultralytics opencv-python numpy pillow
```

### 3. 运行 GUI

```bash
python main.py
```

或使用 Windows 批处理：
```bash
启动营养健康小助手.bat
```

### 4. 训练自己的模型（可选）

```bash
yolo train model=yolov8n.pt data=data.yaml epochs=200 imgsz=640 batch=16
```

### 5. 推理测试

```bash
yolo predict model=best.pt source=your_image.jpg
```

---

## 🏆 项目成果

- **2026 年湖北省职业院校技能大赛 · 人工智能赛道 省级三等奖**
- 作为 4 人团队负责人，统筹赛题拆解、算法方案设计与技术攻坚全流程
- 完整交付数据集处理 → 模型训练调优 → 推理部署验证的赛事交付链路

---

## 🙋 作者说明

本项目由何健伟独立完成全栈开发（需求分析 / 架构设计 / 前后端 / AI 算法 / 训练调优）。

- 作者：何健伟
- GitHub：[@Supu7796](https://github.com/Supu7796)
- 邮箱：1745002884@qq.com

---

## 📜 许可证

MIT License

---

## 🤝 贡献

欢迎提交 Issue 与 Pull Request：
- 提交 bug 请附带复现步骤与日志
- 新增食物类别请同步提供标注样本
