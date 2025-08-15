# file: src/main.py

import sys
import time
from pathlib import Path
import pandas as pd
import json

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent))

from upper_level_optimizer import HeliostatOptimizer
from pysam_interface import test_pysam_interface

def run_dunhuang_optimization(population_size=20, max_generations=10, test_mode=False):
    """
    运行敦煌定日镜场布局多目标优化
    
    Args:
        population_size (int): 种群大小
        max_generations (int): 最大代数
        test_mode (bool): 是否为测试模式（使用更小的参数）
    """
    print("=== 敦煌100MW熔盐塔式光热电站定日镜场布局多目标优化 ===")
    print(f"基于双层协同多目标优化（BL-CMO）框架")
    print(f"上层优化器: H-MOWOA-ABC (混合多目标鲸鱼-人工蜂群算法)")
    print(f"仿真引擎: PySAM SolarPILOT")
    
    if test_mode:
        population_size = 4
        max_generations = 2
        print(f"\n🧪 测试模式: 种群大小={population_size}, 代数={max_generations}")
    else:
        print(f"\n🚀 正式运行: 种群大小={population_size}, 代数={max_generations}")
    
    start_time = time.time()
    
    # 设置气象数据文件路径
    weather_file = Path(__file__).parent.parent / "data" / "raw" / "dunhuang_tmy.csv"
    if not weather_file.exists():
        print(f"❌ 错误: 气象数据文件不存在: {weather_file}")
        return None
        
    print(f"✅ 气象数据文件: {weather_file}")
    
    # 创建优化器
    optimizer = HeliostatOptimizer(
        population_size=population_size,
        max_generations=max_generations,
        weather_file_path=weather_file
    )
    
    try:
        # 运行优化
        final_solutions = optimizer.run()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n=== 优化任务完成 ===")
        print(f"总耗时: {total_time / 60:.2f} 分钟")
        print(f"获得帕累托最优解数量: {len(final_solutions)}")
        
        # 创建结果目录
        results_dir = Path(__file__).parent.parent / "results" / "optimization"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        optimizer.save_results(results_dir)
        
        # 保存简化的CSV结果
        if final_solutions:
            csv_data = []
            for i, solution in enumerate(final_solutions):
                row = {
                    'solution_id': i,
                    'optical_efficiency': -solution.objectives[0],  # 转换回正值
                    'total_cost_usd': solution.objectives[1],
                    'peak_flux_kw_m2': solution.objectives[2],
                    'helio_az_spacing': solution.params.get('helio_az_spacing', 0),
                    'helio_rad_spacing': solution.params.get('helio_rad_spacing', 0)
                }
                csv_data.append(row)
                
            df = pd.DataFrame(csv_data)
            csv_file = results_dir / "pareto_front.csv"
            df.to_csv(csv_file, index=False)
            print(f"✅ CSV结果已保存: {csv_file}")
            
            # 显示最佳解的摘要
            print(f"\n=== 帕累托前沿摘要 ===")
            print(f"光学效率范围: {df['optical_efficiency'].min():.4f} - {df['optical_efficiency'].max():.4f}")
            print(f"总成本范围: ${df['total_cost_usd'].min():.2e} - ${df['total_cost_usd'].max():.2e}")
            print(f"峰值热流范围: {df['peak_flux_kw_m2'].min():.2f} - {df['peak_flux_kw_m2'].max():.2f} kW/m²")
            
        return final_solutions
        
    except Exception as e:
        print(f"❌ 优化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_interface_test():
    """
    运行PySAM接口测试
    """
    print("=== 运行PySAM接口测试 ===")
    
    try:
        results = test_pysam_interface()
        if results:
            print("\n✅ PySAM接口测试通过，可以开始优化任务")
            return True
        else:
            print("\n❌ PySAM接口测试失败，请检查配置")
            return False
    except Exception as e:
        print(f"❌ 接口测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主函数 - 默认运行正式优化模式
    """
    print("=== 敦煌光热电站定日镜场布局优化系统 ===")
    print("基于PySAM的双层协同多目标优化框架")
    
    try:
        # 默认运行正式优化，使用默认参数
        print("\n🚀 开始正式优化任务...")
        print("注意: 这可能需要较长时间 (数小时到数天)")
        
        # 使用默认参数
        pop_size, max_gen = 20, 10
        print(f"使用默认参数: 种群大小={pop_size}, 最大代数={max_gen}")
            
        results = run_dunhuang_optimization(
            population_size=pop_size,
            max_generations=max_gen,
            test_mode=False
        )
        
        if results:
            print("\n🎉 优化任务成功完成!")
            print("结果已保存到 results/optimization/ 目录")
        else:
            print("\n❌ 优化任务失败")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()