# 桌面检测系统 (Table Detection System)

## 项目简介

这是一个基于深度相机的桌面平面检测系统，专门用于检测相机下方的水平桌面。系统使用3D点云处理、RANSAC平面拟合和机器学习技术来准确识别和分析桌面区域。

## 主要功能

- **桌面平面检测**: 从深度图像中检测水平桌面
- **干扰物体去除**: 识别并去除人体、显示器等干扰物体
- **多帧序列处理**: 处理视频序列，提高检测精度和稳定性
- **3D可视化**: 实时显示检测结果和3D点云
- **相机校准**: 支持双目相机校准和深度校正

## 项目结构

```
table/
├── Aurevo-haoyanli/                 # 主要代码目录
│   ├── biaoding/                    # 相机校准相关
│   ├── parameters/                  # 参数配置文件
│   ├── test.py                      # 基础桌面检测器
│   ├── test_2.py                    # 增强版检测器
│   ├── test2new.py                  # 视频序列检测器
│   ├── config.py                    # 配置参数
│   └── main.py                      # 主程序入口
├── 项目结构文档.md                   # 详细项目文档
├── test.py技术文档.md               # 技术文档
└── README.md                        # 项目说明
```

## 安装依赖

```bash
pip install numpy opencv-python open3d matplotlib scipy scikit-learn pandas
```

## 使用方法

### 基础检测

```python
from Aurevo-haoyanli.test import DesktopTableDetector, CONFIG

# 创建检测器
detector = DesktopTableDetector(CONFIG)

# 处理单个文件
result = detector.process_single_file("path/to/depth.npz")
```

### 增强版检测（干扰物体去除）

```python
from Aurevo-haoyanli.test_2 import EnhancedTableDetector

detector = EnhancedTableDetector()
result = detector.process_file("path/to/depth.npz")
```

### 视频序列检测

```python
from Aurevo-haoyanli.test2new import VideoSequenceTableDetector

detector = VideoSequenceTableDetector()
report = detector.process_video_sequence("path/to/sequence/")
```

## 算法特点

1. **多层次检测**:
   - 基于深度梯度的物体边缘检测
   - 深度聚类分割
   - RANSAC平面拟合

2. **干扰物体去除**:
   - 自动识别人体、显示器等干扰物
   - 基于位置和深度特征过滤

3. **多帧融合**:
   - 视频序列处理提高稳定性
   - 异常帧检测和过滤
   - 共识平面计算

## 性能指标

- **精度**: 法向量偏差角度 < 5°
- **稳定性**: 多帧标准差 < 2°
- **处理速度**: ~1-2秒/帧
- **检测成功率**: > 90%

## 配置说明

主要配置参数在 `CONFIG` 字典中：

```python
CONFIG = {
    "MIN_DEPTH_MM": 100,           # 最小深度(毫米)
    "MAX_DEPTH_MM": 700,           # 最大深度(毫米)
    "RANSAC_DISTANCE_THRESHOLD": 0.005,  # RANSAC距离阈值
    "MIN_TABLE_POINTS": 500,       # 最小桌面点数
    # ... 更多配置参数
}
```

## 技术栈

- **Python 3.8+**
- **OpenCV**: 图像处理
- **Open3D**: 3D点云处理
- **NumPy/SciPy**: 数值计算
- **Matplotlib**: 可视化
- **scikit-learn**: 机器学习

## 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

如有问题或建议，请创建 Issue 或联系项目维护者。

## 更新日志

### v1.0.0 (2025-01-31)
- 初始发布
- 基础桌面检测功能
- 增强版干扰物体去除
- 视频序列处理
- 完整的可视化系统