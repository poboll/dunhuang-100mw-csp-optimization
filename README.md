# 敦煌100MW光热电站定日镜场优化数据集

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scientific Data](https://img.shields.io/badge/Journal-Scientific%20Data-green.svg)](https://www.nature.com/sdata/)

本项目为**首航敦煌100MW熔盐塔式光热电站**创建了一个全面的定日镜场布局优化数据集，基于**H-MOWOA-ABC混合多目标优化算法**，用于Scientific Data期刊投稿。

## 🌟 项目亮点

- 🔬 **科学严谨**: 基于真实电站参数和TMY气象数据
- 🧬 **创新算法**: H-MOWOA-ABC混合优化算法 (鲸鱼优化算法 + 人工蜂群算法)
- 📊 **多目标优化**: 同时优化年发电量、LCOE和热通量均匀性
- 🌐 **FAIR原则**: 数据集符合可发现、可访问、可互操作、可重用标准
- 📈 **完整工作流**: 从数据预处理到结果可视化的端到端解决方案

## 🏭 电站信息

- **电站名称**: 首航敦煌100MW熔盐塔式光热电站
- **地理位置**: 甘肃省敦煌市 (40.063°N, 94.426°E, 海拔1267m)
- **装机容量**: 100 MW
- **储热时长**: 11小时
- **塔高**: 263米
- **定日镜数量**: 1000+ (可配置)
- **单镜面积**: 115.7 m²

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd 敦煌光热电站优化

# 安装依赖
pip install -r requirements.txt
```

### 2. 快速测试

```bash
# 运行小规模测试 (推荐首次使用)
python run_optimization.py --test
```

### 3. 完整优化

```bash
# 使用默认配置
python run_optimization.py

# 或使用自定义配置
python run_optimization.py --config my_config.json
```

### 4. 检查环境

```bash
# 仅检查依赖和数据文件
python run_optimization.py --check-only
```

## 项目概述

本项目旨在为《Scientific Data》期刊创建一个符合FAIR原则的定日镜场优化基准数据集，基于首航敦煌100MW熔盐塔式光热电站的真实参数。

## 目录结构

```
.
├── README.md                    # 项目说明文件
├── data/
│   ├── raw/
│   │   ├── tmy.1.csv            # NREL TMY气象数据
│   │   └── GSA_Report_43.431483°, 094.494022° (1).xlsx  # GSA报告数据
│   ├── processed/
│   │   ├── heliostat_layout.csv # 基准定日镜布局
│   │   └── heliostat_layout_optimized.csv # 优化后布局
│   └── spatial/
│       ├── *.tif                # 太阳能资源栅格数据
│       └── *.json               # 地理边界数据
├── src/
│   ├── visualization/
│   │   ├── 1.py                 # DNI地图可视化
│   │   └── 2.py                 # 图像裁剪处理
│   ├── data_processing/
│   │   ├── 3.py                 # 卫星瓦片下载
│   │   ├── 3-1.py               # 瓦片拼接
│   │   └── 3-2.py               # 定日镜检测
│   └── optimization/
│       └── (待添加动物园算法)
├── results/
│   ├── figures/
│   └── layouts/
└── docs/
    ├── data_schema.md
    └── methodology.md
```

## 数据说明

### 气象数据
- **tmy.1.csv**: 敦煌地区典型气象年数据，包含8760小时的太阳辐射和气象参数
- 坐标: 40.063°N, 94.426°E
- 海拔: 1267m

### 空间数据
- **DNI.tif**: 直接法向辐射栅格数据
- **GHI.tif**: 全球水平辐射栅格数据
- **TEMP.tif**: 温度栅格数据
- **xinjiang.json**: 新疆地理边界数据

### 布局数据
- **heliostat_layout.csv**: 从卫星图像检测得到的基准定日镜布局
- **heliostat_layout_optimized.csv**: 经过算法优化的定日镜布局

## 下一步计划

1. **实现动物园算法**: 开发多目标优化算法进行定日镜场布局优化
2. **数据集生成**: 生成帕累托前沿解集
3. **技术验证**: 与实际电站数据对比验证
4. **文档完善**: 编写详细的数据模式和方法论文档

## 引用

如果使用本数据集，请引用相关论文（待发表）。

## 许可证

- 代码: MIT License
- 数据: CC-BY 4.0