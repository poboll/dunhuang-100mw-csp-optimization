#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
敦煌100MW光热电站定日镜场优化 - 快速启动脚本

这是一个简化的启动脚本，用于快速运行优化算法。
适合初次使用或快速测试。

使用方法:
    python run_optimization.py
    
或者指定配置文件:
    python run_optimization.py --config config.json
    
或者运行小规模测试:
    python run_optimization.py --test
"""

import os
import sys
import argparse
import json
from datetime import datetime

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from main import DunhuangHeliostatOptimizer
except ImportError as e:
    print(f"错误: 无法导入主模块 - {e}")
    print("请确保已安装所有依赖包: pip install -r requirements.txt")
    sys.exit(1)

def create_test_config():
    """
    创建测试配置 - 小规模快速测试
    """
    return {
        "plant": {
            "name": "敦煌100MW光热电站 - 测试模式",
            "location": {
                "latitude": 40.063,
                "longitude": 94.426,
                "elevation": 1267
            },
            "tower_position": [0, 0],
            "tower_height": 263,
            "field_bounds": [-400, 400, -400, 400],  # 缩小场地
            "num_heliostats": 100,  # 减少定日镜数量
            "heliostat_size": 115.7,
            "rated_power": 100,
            "storage_hours": 11
        },
        "optimization": {
            "population_size": 20,  # 减小种群
            "max_generations": 30,  # 减少迭代次数
            "algorithm": "H-MOWOA-ABC"
        },
        "objectives": {
            "annual_energy": {"type": "maximize", "weight": 1.0},
            "lcoe": {"type": "minimize", "weight": 1.0},
            "flux_uniformity": {"type": "maximize", "weight": 1.0}
        }
    }

def check_dependencies():
    """
    检查必要的依赖包
    """
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_data_files():
    """
    检查必要的数据文件
    """
    required_files = [
        'data/raw/tmy.1.csv'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  缺少以下数据文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n算法将使用默认参数运行，但可能影响结果准确性。")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='敦煌100MW光热电站定日镜场优化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_optimization.py                    # 使用默认配置
  python run_optimization.py --config my.json  # 使用自定义配置
  python run_optimization.py --test             # 快速测试模式
  python run_optimization.py --check-only       # 仅检查环境
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.json',
        help='配置文件路径 (默认: config.json)'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='运行快速测试模式 (小规模优化)'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='仅检查环境和依赖，不运行优化'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results',
        help='输出目录 (默认: results)'
    )
    
    args = parser.parse_args()
    
    print("🌞 敦煌100MW光热电站定日镜场优化系统")
    print("=" * 50)
    
    # 检查环境
    print("\n🔍 检查运行环境...")
    
    if not check_dependencies():
        return 1
    
    print("✅ Python依赖包检查通过")
    
    data_ok = check_data_files()
    if data_ok:
        print("✅ 数据文件检查通过")
    
    if args.check_only:
        print("\n✅ 环境检查完成")
        return 0
    
    # 准备配置
    if args.test:
        print("\n🧪 使用测试模式配置")
        config = create_test_config()
        config_source = "测试模式"
    elif os.path.exists(args.config):
        print(f"\n📋 加载配置文件: {args.config}")
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        config_source = args.config
    else:
        print(f"\n⚠️  配置文件不存在: {args.config}")
        print("使用默认测试配置")
        config = create_test_config()
        config_source = "默认配置"
    
    # 显示配置信息
    print(f"\n📊 优化参数:")
    print(f"   配置来源: {config_source}")
    print(f"   定日镜数量: {config['plant']['num_heliostats']}")
    print(f"   种群大小: {config['optimization']['population_size']}")
    print(f"   迭代次数: {config['optimization']['max_generations']}")
    print(f"   场地范围: {config['plant']['field_bounds']}")
    
    # 估算运行时间
    estimated_time = (
        config['optimization']['population_size'] * 
        config['optimization']['max_generations'] * 
        config['plant']['num_heliostats'] / 10000
    )
    print(f"   预估耗时: {estimated_time:.1f} 分钟")
    
    # 确认运行
    if not args.test:
        response = input("\n是否开始优化? (y/N): ")
        if response.lower() not in ['y', 'yes', '是']:
            print("已取消")
            return 0
    
    print("\n🚀 开始优化...")
    start_time = datetime.now()
    
    try:
        # 创建临时配置文件
        temp_config_file = 'temp_config.json'
        with open(temp_config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 创建优化器并运行
        optimizer = DunhuangHeliostatOptimizer(temp_config_file)
        results, analysis = optimizer.run_complete_workflow()
        
        # 清理临时文件
        if os.path.exists(temp_config_file):
            os.remove(temp_config_file)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds() / 60
        
        if results is not None:
            print(f"\n🎉 优化成功完成! (耗时: {total_time:.1f} 分钟)")
            
            print(f"\n📈 主要结果:")
            print(f"   帕累托前沿解: {len(results['pareto_front'])} 个")
            
            if analysis:
                best = analysis['best_compromise_solution']
                print(f"   最佳年发电量: {best['annual_energy']:.1f} MWh")
                print(f"   最佳LCOE: {best['lcoe']:.1f} $/MWh")
                print(f"   最佳热通量均匀性: {best['flux_uniformity']:.3f}")
            
            print(f"\n📁 输出文件位置:")
            results_path = os.path.abspath(optimizer.results_dir)
            print(f"   {results_path}")
            
            print(f"\n🔗 主要文件:")
            print(f"   • 帕累托前沿: pareto_front.csv")
            print(f"   • 最优布局: optimal_layouts/")
            print(f"   • 可视化图表: figures/")
            print(f"   • 数据集摘要: dataset_summary.md")
            
            if args.test:
                print(f"\n💡 提示: 这是测试模式结果")
                print(f"   如需完整优化，请运行: python run_optimization.py")
            
            return 0
        else:
            print(f"\n❌ 优化失败")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️  用户中断优化")
        return 1
    except Exception as e:
        print(f"\n❌ 运行错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)