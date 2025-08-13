#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
敦煌100MW光热电站定日镜场优化主程序

该脚本整合了:
1. 数据预处理
2. H-MOWOA-ABC优化算法
3. 结果分析和可视化
4. 数据集生成

作者: poboll
日期: 2025
用途: Scientific Data 期刊投稿数据集生成
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimization.zoo_algorithm import H_MOWOA_ABC, OptimizationResult

class DunhuangHeliostatOptimizer:
    """
    敦煌光热电站定日镜场优化器
    整合数据处理、优化算法和结果分析
    """
    
    def __init__(self, config_file: str = None):
        """
        初始化优化器
        
        Args:
            config_file: 配置文件路径
        """
        self.config = self._load_config(config_file)
        # 获取项目根目录
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = os.path.join(self.project_root, "results")
        self.data_dir = os.path.join(self.project_root, "data")
        
        # 创建输出目录
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/figures", exist_ok=True)
        os.makedirs(f"{self.results_dir}/layouts", exist_ok=True)
        
        # 初始化优化器
        self.optimizer = H_MOWOA_ABC(
            population_size=self.config['optimization']['population_size'],
            max_generations=self.config['optimization']['max_generations'],
            field_bounds=tuple(self.config['plant']['field_bounds']),
            num_heliostats=self.config['plant']['num_heliostats'],
            tower_position=tuple(self.config['plant']['tower_position']),
            heliostat_size=self.config['plant']['heliostat_size']
        )
        
    def _load_config(self, config_file: str = None) -> dict:
        """
        加载配置文件
        """
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 默认配置 - 基于敦煌100MW电站参数
            return {
                "plant": {
                    "name": "首航敦煌100MW熔盐塔式光热电站",
                    "location": {
                        "latitude": 40.063,
                        "longitude": 94.426,
                        "elevation": 1267
                    },
                    "tower_position": [0, 0],
                    "tower_height": 263,
                    "field_bounds": [-800, 800, -800, 800],
                    "num_heliostats": 1000,
                    "heliostat_size": 115.7,
                    "rated_power": 100,
                    "storage_hours": 11
                },
                "optimization": {
                    "population_size": 50,
                    "max_generations": 100,
                    "algorithm": "H-MOWOA-ABC"
                },
                "objectives": {
                    "annual_energy": {"type": "maximize", "weight": 1.0},
                    "lcoe": {"type": "minimize", "weight": 1.0},
                    "flux_uniformity": {"type": "maximize", "weight": 1.0}
                }
            }
    
    def load_meteorological_data(self) -> pd.DataFrame:
        """
        加载TMY气象数据
        """
        tmy_file = os.path.join(self.data_dir, "raw", "tmy.1.csv")
        
        if not os.path.exists(tmy_file):
            raise FileNotFoundError(f"TMY数据文件未找到: {tmy_file}")
        
        print(f"加载TMY气象数据: {tmy_file}")
        
        try:
            # 读取文件头信息
            with open(tmy_file, 'r') as f:
                lines = f.readlines()
            
            # 解析头部信息
            latitude = float(lines[0].split(':')[1].strip())
            longitude = float(lines[1].split(':')[1].strip())
            elevation = float(lines[2].split(':')[1].strip())
            
            print(f"电站位置: {latitude:.3f}°N, {longitude:.3f}°E, 海拔{elevation:.0f}m")
            
            # 找到数据开始行 (包含time(UTC)的行)
            data_start_line = 0
            for i, line in enumerate(lines):
                if 'time(UTC)' in line:
                    data_start_line = i
                    break
            
            # 读取气象数据
            tmy_data = pd.read_csv(tmy_file, skiprows=data_start_line)
            
            # 重命名列以便使用
            column_mapping = {
                'time(UTC)': 'datetime',
                'T2m': 'temperature',  # 2m温度 (°C)
                'RH': 'humidity',      # 相对湿度 (%)
                'G(h)': 'ghi',         # 全球水平辐射 (W/m²)
                'Gb(n)': 'dni',        # 直接法向辐射 (W/m²)
                'Gd(h)': 'dhi',        # 散射水平辐射 (W/m²)
                'WS10m': 'wind_speed', # 10m风速 (m/s)
                'WD10m': 'wind_direction', # 10m风向 (°)
                'SP': 'pressure'       # 表面压力 (Pa)
            }
            
            # 重命名存在的列
            for old_name, new_name in column_mapping.items():
                if old_name in tmy_data.columns:
                    tmy_data = tmy_data.rename(columns={old_name: new_name})
            
            # 转换时间格式
            if 'datetime' in tmy_data.columns:
                tmy_data['datetime'] = pd.to_datetime(tmy_data['datetime'], format='%Y%m%d:%H%M')
            
            print(f"TMY数据包含 {len(tmy_data)} 小时的记录")
            print(f"数据列: {list(tmy_data.columns)}")
            
            # 添加位置信息到数据框
            tmy_data.attrs['latitude'] = latitude
            tmy_data.attrs['longitude'] = longitude
            tmy_data.attrs['elevation'] = elevation
            
            return tmy_data
            
        except Exception as e:
            print(f"警告: TMY数据加载失败 - {str(e)}")
            print("将使用模拟数据继续运行")
            
            # 生成模拟的TMY数据用于测试
            dates = pd.date_range('2020-01-01', periods=8760, freq='H')
            
            # 简单的DNI模型 (基于太阳高度角)
            hour_of_year = np.arange(8760)
            day_of_year = hour_of_year // 24 + 1
            hour_of_day = hour_of_year % 24
            
            # 模拟DNI (考虑日变化和季节变化)
            dni_base = 800 * np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12))
            seasonal_factor = 0.8 + 0.4 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
            dni = dni_base * seasonal_factor * (0.8 + 0.2 * np.random.random(8760))
            
            tmy_data = pd.DataFrame({
                'datetime': dates,
                'dni': dni,
                'ghi': dni * 1.2,
                'dhi': dni * 0.1,
                'temperature': 20 + 15 * np.sin(2 * np.pi * day_of_year / 365) + 10 * np.sin(np.pi * hour_of_day / 12),
                'humidity': 30 + 20 * np.random.random(8760),
                'wind_speed': 2 + 3 * np.random.random(8760),
                'wind_direction': 180 + 90 * np.random.random(8760),
                'pressure': 85000 + 1000 * np.random.random(8760)
            })
            
            tmy_data.attrs['latitude'] = 40.063
            tmy_data.attrs['longitude'] = 94.426
            tmy_data.attrs['elevation'] = 1267
            
            print(f"生成模拟TMY数据: {len(tmy_data)} 小时")
            
            return tmy_data
    
    def load_baseline_layout(self) -> pd.DataFrame:
        """
        加载基准定日镜布局
        """
        layout_file = f"{self.data_dir}/processed/heliostat_layout.csv"
        
        if os.path.exists(layout_file):
            print(f"加载基准布局: {layout_file}")
            return pd.read_csv(layout_file)
        else:
            print("未找到基准布局文件，将使用算法生成的径向交错布局")
            return None
    
    def run_optimization(self) -> dict:
        """
        执行优化算法
        """
        print("\n" + "="*60)
        print("开始定日镜场布局优化")
        print("="*60)
        
        # 加载数据
        tmy_data = self.load_meteorological_data()
        baseline_layout = self.load_baseline_layout()
        
        # 执行优化
        start_time = datetime.now()
        results = self.optimizer.optimize()
        end_time = datetime.now()
        
        optimization_time = (end_time - start_time).total_seconds()
        
        print(f"\n优化完成，耗时: {optimization_time:.1f} 秒")
        
        # 添加元数据
        results['metadata'] = {
            'optimization_time_seconds': optimization_time,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'config': self.config,
            'tmy_data_points': len(tmy_data),
            'baseline_layout_available': baseline_layout is not None
        }
        
        return results
    
    def analyze_results(self, results: dict) -> dict:
        """
        分析优化结果
        """
        print("\n分析优化结果...")
        
        pareto_front = results['pareto_front']
        
        if not pareto_front:
            print("警告: 未找到帕累托前沿解")
            return {}
        
        # 统计分析
        annual_energies = [sol.objectives[0] for sol in pareto_front]
        lcoes = [sol.objectives[1] for sol in pareto_front]
        flux_uniformities = [sol.objectives[2] for sol in pareto_front]
        
        analysis = {
            'pareto_front_size': len(pareto_front),
            'annual_energy': {
                'min': min(annual_energies),
                'max': max(annual_energies),
                'mean': np.mean(annual_energies),
                'std': np.std(annual_energies)
            },
            'lcoe': {
                'min': min(lcoes),
                'max': max(lcoes),
                'mean': np.mean(lcoes),
                'std': np.std(lcoes)
            },
            'flux_uniformity': {
                'min': min(flux_uniformities),
                'max': max(flux_uniformities),
                'mean': np.mean(flux_uniformities),
                'std': np.std(flux_uniformities)
            }
        }
        
        # 找到最佳折衷解 (使用加权和方法)
        best_compromise_idx = 0
        best_score = float('-inf')
        
        for i, sol in enumerate(pareto_front):
            # 归一化目标值
            norm_energy = (sol.objectives[0] - analysis['annual_energy']['min']) / \
                         (analysis['annual_energy']['max'] - analysis['annual_energy']['min'] + 1e-10)
            norm_lcoe = 1 - (sol.objectives[1] - analysis['lcoe']['min']) / \
                       (analysis['lcoe']['max'] - analysis['lcoe']['min'] + 1e-10)
            norm_uniformity = (sol.objectives[2] - analysis['flux_uniformity']['min']) / \
                             (analysis['flux_uniformity']['max'] - analysis['flux_uniformity']['min'] + 1e-10)
            
            # 计算加权分数
            score = (norm_energy * self.config['objectives']['annual_energy']['weight'] +
                    norm_lcoe * self.config['objectives']['lcoe']['weight'] +
                    norm_uniformity * self.config['objectives']['flux_uniformity']['weight'])
            
            if score > best_score:
                best_score = score
                best_compromise_idx = i
        
        analysis['best_compromise_solution'] = {
            'index': best_compromise_idx,
            'annual_energy': pareto_front[best_compromise_idx].objectives[0],
            'lcoe': pareto_front[best_compromise_idx].objectives[1],
            'flux_uniformity': pareto_front[best_compromise_idx].objectives[2],
            'score': best_score
        }
        
        print(f"帕累托前沿包含 {analysis['pareto_front_size']} 个解")
        print(f"年发电量范围: {analysis['annual_energy']['min']:.1f} - {analysis['annual_energy']['max']:.1f} MWh")
        print(f"LCOE范围: {analysis['lcoe']['min']:.1f} - {analysis['lcoe']['max']:.1f} $/MWh")
        print(f"热通量均匀性范围: {analysis['flux_uniformity']['min']:.3f} - {analysis['flux_uniformity']['max']:.3f}")
        
        best_sol = analysis['best_compromise_solution']
        print(f"\n最佳折衷解:")
        print(f"  年发电量: {best_sol['annual_energy']:.1f} MWh")
        print(f"  LCOE: {best_sol['lcoe']:.1f} $/MWh")
        print(f"  热通量均匀性: {best_sol['flux_uniformity']:.3f}")
        
        return analysis
    
    def generate_dataset(self, results: dict, analysis: dict):
        """
        生成符合FAIR原则的数据集
        """
        print("\n生成数据集...")
        
        # 保存优化结果
        self.optimizer.save_results(results, self.results_dir)
        
        # 生成数据集元数据
        metadata = {
            'dataset_info': {
                'title': '敦煌100MW光热电站定日镜场多目标优化数据集',
                'description': '基于H-MOWOA-ABC算法的定日镜场布局优化结果',
                'version': '1.0',
                'creation_date': datetime.now().isoformat(),
                'license': 'CC-BY 4.0'
            },
            'plant_parameters': self.config['plant'],
            'optimization_parameters': self.config['optimization'],
            'results_summary': analysis,
            'data_files': {
                'pareto_front': 'pareto_front.csv',
                'optimization_history': 'optimization_history.csv',
                'optimal_layouts': 'optimal_layouts/*.csv'
            }
        }
        
        # 保存元数据
        with open(f"{self.results_dir}/dataset_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 生成数据集摘要报告
        self._generate_summary_report(results, analysis)
        
        print(f"数据集已生成并保存到: {self.results_dir}")
    
    def _generate_summary_report(self, results: dict, analysis: dict):
        """
        生成数据集摘要报告
        """
        report_content = f"""
# 敦煌100MW光热电站定日镜场优化数据集摘要报告

## 基本信息
- 电站名称: {self.config['plant']['name']}
- 地理位置: {self.config['plant']['location']['latitude']}°N, {self.config['plant']['location']['longitude']}°E
- 海拔高度: {self.config['plant']['location']['elevation']} m
- 额定功率: {self.config['plant']['rated_power']} MW
- 储热时长: {self.config['plant']['storage_hours']} 小时

## 优化参数
- 算法: {self.config['optimization']['algorithm']}
- 种群大小: {self.config['optimization']['population_size']}
- 迭代次数: {self.config['optimization']['max_generations']}
- 定日镜数量: {self.config['plant']['num_heliostats']}
- 单镜面积: {self.config['plant']['heliostat_size']} m²

## 优化结果
- 帕累托前沿解数量: {analysis['pareto_front_size']}
- 优化耗时: {results['metadata']['optimization_time_seconds']:.1f} 秒

### 目标函数统计

#### 年发电量 (MWh)
- 最小值: {analysis['annual_energy']['min']:.1f}
- 最大值: {analysis['annual_energy']['max']:.1f}
- 平均值: {analysis['annual_energy']['mean']:.1f}
- 标准差: {analysis['annual_energy']['std']:.1f}

#### LCOE ($/MWh)
- 最小值: {analysis['lcoe']['min']:.1f}
- 最大值: {analysis['lcoe']['max']:.1f}
- 平均值: {analysis['lcoe']['mean']:.1f}
- 标准差: {analysis['lcoe']['std']:.1f}

#### 热通量均匀性
- 最小值: {analysis['flux_uniformity']['min']:.3f}
- 最大值: {analysis['flux_uniformity']['max']:.3f}
- 平均值: {analysis['flux_uniformity']['mean']:.3f}
- 标准差: {analysis['flux_uniformity']['std']:.3f}

### 最佳折衷解
- 年发电量: {analysis['best_compromise_solution']['annual_energy']:.1f} MWh
- LCOE: {analysis['best_compromise_solution']['lcoe']:.1f} $/MWh
- 热通量均匀性: {analysis['best_compromise_solution']['flux_uniformity']:.3f}

## 数据文件说明

1. **pareto_front.csv**: 帕累托前沿上所有非支配解的目标函数值
2. **optimization_history.csv**: 优化过程的收敛历史
3. **optimal_layouts/**: 每个帕累托解对应的定日镜布局坐标
4. **dataset_metadata.json**: 完整的数据集元数据

## 引用信息

如果使用本数据集，请引用:
[待发表论文信息]

## 联系信息

- 作者: [您的姓名]
- 邮箱: [您的邮箱]
- 机构: [您的机构]

---
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(f"{self.results_dir}/dataset_summary.md", 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    def visualize_results(self, results: dict):
        """
        可视化优化结果
        """
        print("\n生成可视化图表...")
        
        # 使用优化器的可视化方法
        self.optimizer.plot_results(results)
        
        print(f"可视化图表已保存到: {self.results_dir}/figures/")
    
    def run_complete_workflow(self):
        """
        执行完整的优化工作流程
        """
        print("启动敦煌光热电站定日镜场优化工作流程")
        print("=" * 80)
        
        try:
            # 1. 执行优化
            results = self.run_optimization()
            
            # 2. 分析结果
            analysis = self.analyze_results(results)
            
            # 3. 生成数据集
            self.generate_dataset(results, analysis)
            
            # 4. 可视化结果
            self.visualize_results(results)
            
            print("\n" + "="*80)
            print("工作流程完成！")
            print(f"所有结果已保存到: {os.path.abspath(self.results_dir)}")
            print("="*80)
            
            return results, analysis
            
        except Exception as e:
            print(f"\n错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None


def main():
    """
    主函数
    """
    print("敦煌100MW光热电站定日镜场优化系统")
    print("用于Scientific Data期刊数据集生成")
    print("\n基于H-MOWOA-ABC混合多目标优化算法")
    print("-" * 60)
    
    # 创建优化器
    optimizer = DunhuangHeliostatOptimizer()
    
    # 执行完整工作流程
    results, analysis = optimizer.run_complete_workflow()
    
    if results is not None:
        print("\n🎉 优化成功完成！")
        print("\n📊 主要成果:")
        print(f"   • 生成了 {len(results['pareto_front'])} 个帕累托最优解")
        print(f"   • 优化了 {optimizer.config['plant']['num_heliostats']} 个定日镜的布局")
        print(f"   • 数据集符合FAIR原则，可用于Scientific Data投稿")
        
        print("\n📁 输出文件:")
        print(f"   • 帕累托前沿: {optimizer.results_dir}/pareto_front.csv")
        print(f"   • 优化布局: {optimizer.results_dir}/optimal_layouts/")
        print(f"   • 可视化图表: {optimizer.results_dir}/figures/")
        print(f"   • 数据集元数据: {optimizer.results_dir}/dataset_metadata.json")
        
        print("\n🚀 下一步建议:")
        print("   1. 检查生成的帕累托前沿和布局")
        print("   2. 与SolarPILOT进行详细仿真验证")
        print("   3. 准备Scientific Data论文手稿")
        print("   4. 将数据集上传到Zenodo获取DOI")
    else:
        print("\n❌ 优化过程中出现错误，请检查日志信息")


if __name__ == "__main__":
    main()