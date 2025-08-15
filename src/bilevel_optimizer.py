#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双层协同多目标优化器 (BL-CMO) - 主控制器

集成上层优化器 (H-MOWOA-ABC) 和下层优化器 (CMA-ES)
实现完整的双层协同多目标优化框架

上层: 优化定日镜场布局参数 (间距、排列等)
下层: 优化详细设计参数 (镜面尺寸、光学参数等)

作者: AI Assistant
日期: 2024
"""

import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import matplotlib.pyplot as plt
from upper_level_optimizer import HeliostatOptimizer, Individual
from lower_level_optimizer import CMAESOptimizer
from pysam_interface import evaluate_fitness

class BilevelOptimizer:
    """双层协同多目标优化器"""
    
    def __init__(self, 
                 upper_population_size=20,
                 upper_generations=10,
                 lower_evaluations=50,
                 weather_file_path=None,
                 output_dir=None):
        """
        初始化双层优化器
        
        Args:
            upper_population_size: 上层种群大小
            upper_generations: 上层优化代数
            lower_evaluations: 下层优化评估次数
            weather_file_path: 气象数据文件路径
            output_dir: 输出目录
        """
        self.upper_population_size = upper_population_size
        self.upper_generations = upper_generations
        self.lower_evaluations = lower_evaluations
        self.weather_file_path = weather_file_path
        self.output_dir = Path(output_dir) if output_dir else Path("results/bilevel_optimization")
        
        # 统计信息
        self.total_evaluations = 0
        self.optimization_history = []
        self.final_solutions = []
        
    def _enhanced_fitness_evaluation(self, layout_params: Dict[str, float]) -> Dict:
        """
        增强的适应度评估函数
        对每个上层个体，运行下层CMA-ES优化获得最佳详细参数
        
        Args:
            layout_params: 上层布局参数
            
        Returns:
            包含目标函数值和最佳下层参数的字典
        """
        print(f"🔄 双层评估: 上层参数 {layout_params}")
        
        # 创建下层优化器
        lower_optimizer = CMAESOptimizer(
            upper_params=layout_params,
            max_evaluations=self.lower_evaluations
        )
        
        # 执行下层优化
        best_lower_solution = lower_optimizer.optimize()
        
        if best_lower_solution is None:
            print("❌ 下层优化失败，使用默认评估")
            return evaluate_fitness(layout_params)
        
        # 更新总评估次数
        self.total_evaluations += lower_optimizer.evaluations
        
        # 返回下层优化的最佳结果
        result = best_lower_solution['objectives'].copy()
        result['lower_params'] = {k: v for k, v in best_lower_solution['params'].items() 
                                 if k not in layout_params}
        result['lower_fitness'] = best_lower_solution['fitness']
        result['lower_evaluations'] = lower_optimizer.evaluations
        
        print(f"✅ 双层评估完成: 效率={result['f1_eff']:.4f}, 成本=${result['f2_cost']:.2e}, 热流={result['f3_flux']:.2f}")
        
        return result
    
    def optimize(self) -> List[Dict]:
        """
        执行双层协同多目标优化
        
        Returns:
            帕累托最优解列表
        """
        print("🚀 启动双层协同多目标优化 (BL-CMO)")
        print(f"   上层: H-MOWOA-ABC (种群={self.upper_population_size}, 代数={self.upper_generations})")
        print(f"   下层: CMA-ES (评估次数={self.lower_evaluations})")
        print(f"   输出目录: {self.output_dir}")
        
        start_time = time.time()
        
        # 创建上层优化器
        upper_optimizer = HeliostatOptimizer(
            population_size=self.upper_population_size,
            max_generations=self.upper_generations,
            weather_file_path=self.weather_file_path
        )
        
        # 修改上层优化器的评估方法
        # 保存原始的_evaluate_population方法
        original_evaluate_population = upper_optimizer._evaluate_population
        
        # 创建新的评估方法
        def enhanced_evaluate_population():
            print(f"🔄 评估第 {upper_optimizer.generation + 1} 代种群适应度 (双层模式)")
            
            unevaluated_count = 0
            for individual in upper_optimizer.population:
                if individual.objectives is None:
                    unevaluated_count += 1
                    
            print(f"   需要评估的个体数量: {unevaluated_count}")
            
            for i, individual in enumerate(upper_optimizer.population):
                if individual.objectives is None:  # 只评估新个体
                    start_time = time.time()
                    
                    # 使用双层评估函数
                    results = self._enhanced_fitness_evaluation(individual.params)
                    
                    eval_time = time.time() - start_time
                    upper_optimizer.history['evaluation_times'].append(eval_time)
                    
                    if results:
                        # 设置目标函数值 (注意：效率需要取负值用于最小化)
                        individual.objectives = [
                            -results['f1_eff'],  # 最大化效率 -> 最小化负效率
                            results['f2_cost'],   # 最小化成本
                            results['f3_flux']    # 最小化峰值热流
                        ]
                        
                        # 保存下层优化结果
                        if 'lower_params' in results:
                            individual.lower_params = results['lower_params']
                        if 'lower_fitness' in results:
                            individual.lower_fitness = results['lower_fitness']
                        if 'lower_evaluations' in results:
                            individual.lower_evaluations = results['lower_evaluations']
                    else:
                        # 评估失败，设置惩罚值
                        individual.objectives = [0.0, float('inf'), float('inf')]
                    
                    print(f"   个体 {i+1}/{len(upper_optimizer.population)} 评估完成 (耗时: {eval_time:.2f}s)")
            
            upper_optimizer.evaluation_count += unevaluated_count
            print(f"✅ 种群评估完成，累计评估次数: {upper_optimizer.evaluation_count}")
        
        # 替换评估方法
        upper_optimizer._evaluate_population = enhanced_evaluate_population
        
        # 执行上层优化
        print("\n=== 开始上层优化 ===")
        pareto_solutions = upper_optimizer.run()
        
        # 处理最终解
        self.final_solutions = []
        for i, individual in enumerate(pareto_solutions):
            solution = {
                'solution_id': i,
                'upper_params': individual.params.copy(),
                'objectives': {
                    'optical_efficiency': individual.objectives[0] if individual.objectives[0] < 0 else -individual.objectives[0],
                    'total_cost_usd': individual.objectives[1],
                    'peak_flux_kw_m2': individual.objectives[2]
                },
                'lower_params': getattr(individual, 'lower_params', {}),
                'lower_fitness': getattr(individual, 'lower_fitness', None),
                'lower_evaluations': getattr(individual, 'lower_evaluations', 0)
            }
            self.final_solutions.append(solution)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 双层协同优化完成!")
        print(f"   总耗时: {total_time/60:.2f} 分钟")
        print(f"   总评估次数: {self.total_evaluations}")
        print(f"   帕累托最优解数量: {len(self.final_solutions)}")
        
        # 保存结果
        self.save_results()
        
        return self.final_solutions
    
    def save_results(self):
        """保存优化结果"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 保存双层优化结果到: {self.output_dir}")
        
        # 保存帕累托前沿 (JSON格式)
        with open(self.output_dir / 'bilevel_pareto_front.json', 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'algorithm': 'BL-CMO (H-MOWOA-ABC + CMA-ES)',
                    'upper_population_size': self.upper_population_size,
                    'upper_generations': self.upper_generations,
                    'lower_evaluations': self.lower_evaluations,
                    'total_evaluations': self.total_evaluations,
                    'solution_count': len(self.final_solutions)
                },
                'solutions': self.final_solutions
            }, f, indent=2, ensure_ascii=False)
        
        # 保存帕累托前沿 (CSV格式)
        import pandas as pd
        
        csv_data = []
        for sol in self.final_solutions:
            row = {
                'solution_id': sol['solution_id'],
                'optical_efficiency': sol['objectives']['optical_efficiency'],
                'total_cost_usd': sol['objectives']['total_cost_usd'],
                'peak_flux_kw_m2': sol['objectives']['peak_flux_kw_m2'],
                'lower_fitness': sol['lower_fitness'],
                'lower_evaluations': sol['lower_evaluations']
            }
            # 添加上层参数
            for k, v in sol['upper_params'].items():
                row[f'upper_{k}'] = v
            # 添加下层参数
            for k, v in sol['lower_params'].items():
                row[f'lower_{k}'] = v
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(self.output_dir / 'bilevel_pareto_front.csv', index=False)
        
        # 生成可视化图表
        self._create_visualizations()
        
        print(f"✅ 结果保存完成")
        print(f"   - JSON: bilevel_pareto_front.json")
        print(f"   - CSV: bilevel_pareto_front.csv")
        print(f"   - 图表: bilevel_objectives_3d.png, bilevel_comparison.png")
    
    def _create_visualizations(self):
        """创建可视化图表"""
        if not self.final_solutions:
            return
        
        # 提取目标函数值
        efficiencies = [sol['objectives']['optical_efficiency'] for sol in self.final_solutions]
        costs = [sol['objectives']['total_cost_usd'] for sol in self.final_solutions]
        fluxes = [sol['objectives']['peak_flux_kw_m2'] for sol in self.final_solutions]
        
        # 3D帕累托前沿图
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(efficiencies, costs, fluxes, 
                           c=range(len(efficiencies)), cmap='viridis', s=50, alpha=0.7)
        
        ax.set_xlabel('光学效率')
        ax.set_ylabel('总成本 (USD)')
        ax.set_zlabel('峰值热流密度 (kW/m²)')
        ax.set_title('双层协同优化 - 3D帕累托前沿')
        
        plt.colorbar(scatter, label='解编号')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bilevel_objectives_3d.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 对比分析图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 效率 vs 成本
        axes[0, 0].scatter(efficiencies, costs, alpha=0.7, c='blue')
        axes[0, 0].set_xlabel('光学效率')
        axes[0, 0].set_ylabel('总成本 (USD)')
        axes[0, 0].set_title('效率 vs 成本')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 效率 vs 热流
        axes[0, 1].scatter(efficiencies, fluxes, alpha=0.7, c='red')
        axes[0, 1].set_xlabel('光学效率')
        axes[0, 1].set_ylabel('峰值热流密度 (kW/m²)')
        axes[0, 1].set_title('效率 vs 热流密度')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 成本 vs 热流
        axes[1, 0].scatter(costs, fluxes, alpha=0.7, c='green')
        axes[1, 0].set_xlabel('总成本 (USD)')
        axes[1, 0].set_ylabel('峰值热流密度 (kW/m²)')
        axes[1, 0].set_title('成本 vs 热流密度')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 下层评估次数分布
        lower_evals = [sol['lower_evaluations'] for sol in self.final_solutions if sol['lower_evaluations'] > 0]
        if lower_evals:
            axes[1, 1].hist(lower_evals, bins=10, alpha=0.7, color='orange')
            axes[1, 1].set_xlabel('下层评估次数')
            axes[1, 1].set_ylabel('解的数量')
            axes[1, 1].set_title('下层优化评估次数分布')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bilevel_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_summary_statistics(self) -> Dict:
        """获取优化结果摘要统计"""
        if not self.final_solutions:
            return {}
        
        efficiencies = [sol['objectives']['optical_efficiency'] for sol in self.final_solutions]
        costs = [sol['objectives']['total_cost_usd'] for sol in self.final_solutions]
        fluxes = [sol['objectives']['peak_flux_kw_m2'] for sol in self.final_solutions]
        
        return {
            'solution_count': len(self.final_solutions),
            'total_evaluations': self.total_evaluations,
            'efficiency_range': [min(efficiencies), max(efficiencies)],
            'cost_range': [min(costs), max(costs)],
            'flux_range': [min(fluxes), max(fluxes)],
            'avg_efficiency': np.mean(efficiencies),
            'avg_cost': np.mean(costs),
            'avg_flux': np.mean(fluxes)
        }

def test_bilevel_optimizer():
    """测试双层协同优化器"""
    print("=== 测试双层协同多目标优化器 ===")
    
    # 创建优化器 (使用较小的参数进行快速测试)
    optimizer = BilevelOptimizer(
        upper_population_size=4,
        upper_generations=3,
        lower_evaluations=10,
        output_dir="results/test_bilevel"
    )
    
    # 执行优化
    solutions = optimizer.optimize()
    
    if solutions:
        print("\n✅ 双层优化器测试成功!")
        
        # 显示摘要统计
        stats = optimizer.get_summary_statistics()
        print("\n📊 优化结果摘要:")
        for key, value in stats.items():
            if isinstance(value, list):
                print(f"   {key}: [{value[0]:.4f}, {value[1]:.4f}]")
            elif isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        return solutions
    else:
        print("\n❌ 双层优化器测试失败!")
        return None

if __name__ == "__main__":
    test_bilevel_optimizer()