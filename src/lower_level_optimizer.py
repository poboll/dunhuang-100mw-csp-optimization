#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下层优化器 (CMA-ES) - 双层协同多目标优化框架

基于CMA-ES (Covariance Matrix Adaptation Evolution Strategy) 算法
用于在给定上层布局参数的情况下，优化定日镜场的详细配置参数

作者: AI Assistant
日期: 2024
"""

import numpy as np
import random
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
from pysam_interface import evaluate_fitness

class CMAESOptimizer:
    """CMA-ES下层优化器"""
    
    def __init__(self, upper_params: Dict[str, float], max_evaluations=100):
        """
        初始化CMA-ES优化器
        
        Args:
            upper_params: 上层优化器传递的布局参数
            max_evaluations: 最大评估次数
        """
        self.upper_params = upper_params
        self.max_evaluations = max_evaluations
        
        # 下层优化的参数边界 (详细配置参数)
        self.bounds = {
            'helio_width': (10.0, 15.0),           # 定日镜宽度 (m)
            'helio_height': (10.0, 15.0),          # 定日镜高度 (m)
            'helio_optical_error': (0.001, 0.01),  # 光学误差
            'helio_reflectance': (0.85, 0.95),     # 反射率
            'rec_height_spec': (15.0, 30.0),       # 吸热器物理高度 (m)
            'rec_width': (12.0, 25.0),             # 吸热器直径 (m)
            'rec_absorptance': (0.85, 0.98),       # 吸热器吸收率
        }
        
        # CMA-ES参数
        self.dimension = len(self.bounds)
        self.population_size = 4 + int(3 * np.log(self.dimension))  # λ
        self.mu = self.population_size // 2  # μ
        
        # 初始化CMA-ES状态
        self._initialize_cmaes()
        
        # 统计信息
        self.evaluations = 0
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []
        
    def _initialize_cmaes(self):
        """初始化CMA-ES算法参数"""
        # 初始均值 (参数空间中心)
        self.mean = np.array([np.mean(bounds) for bounds in self.bounds.values()])
        
        # 初始步长
        self.sigma = 0.3
        
        # 协方差矩阵
        self.C = np.eye(self.dimension)
        
        # 权重
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights**2)
        
        # 学习率
        self.cc = (4 + self.mu_eff/self.dimension) / (self.dimension + 4 + 2*self.mu_eff/self.dimension)
        self.cs = (self.mu_eff + 2) / (self.dimension + self.mu_eff + 5)
        self.c1 = 2 / ((self.dimension + 1.3)**2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1/self.mu_eff) / ((self.dimension + 2)**2 + self.mu_eff))
        self.damps = 1 + 2*max(0, np.sqrt((self.mu_eff-1)/(self.dimension+1)) - 1) + self.cs
        
        # 进化路径
        self.pc = np.zeros(self.dimension)
        self.ps = np.zeros(self.dimension)
        
        # 期望值
        self.chiN = np.sqrt(self.dimension) * (1 - 1/(4*self.dimension) + 1/(21*self.dimension**2))
        
    def _params_to_vector(self, params: Dict[str, float]) -> np.ndarray:
        """将参数字典转换为向量"""
        return np.array([params[key] for key in self.bounds.keys()])
    
    def _vector_to_params(self, vector: np.ndarray) -> Dict[str, float]:
        """将向量转换为参数字典"""
        params = {}
        for i, key in enumerate(self.bounds.keys()):
            # 边界约束
            min_val, max_val = self.bounds[key]
            params[key] = np.clip(vector[i], min_val, max_val)
        return params
    
    def _evaluate_individual(self, params: Dict[str, float]) -> float:
        """评估单个个体的适应度"""
        # 合并上层和下层参数
        full_params = {**self.upper_params, **params}
        
        # 调用PySAM评估
        results = evaluate_fitness(full_params)
        
        if results is None:
            return float('inf')
        
        # 多目标加权组合 (可根据需要调整权重)
        # 目标: 最大化效率，最小化成本，最小化峰值热流
        w1, w2, w3 = 0.4, 0.4, 0.2  # 权重
        
        # 归一化目标函数
        eff_norm = 1.0 - results['f1_eff']  # 转换为最小化
        cost_norm = results['f2_cost'] / 1e8  # 归一化成本
        flux_norm = results['f3_flux'] / 1000  # 归一化热流
        
        fitness = w1 * eff_norm + w2 * cost_norm + w3 * flux_norm
        
        self.evaluations += 1
        
        # 更新最佳解
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = {
                'params': full_params.copy(),
                'fitness': fitness,
                'objectives': results
            }
        
        return fitness
    
    def optimize(self) -> Dict:
        """执行CMA-ES优化"""
        print(f"🔄 启动CMA-ES下层优化")
        print(f"   上层参数: {self.upper_params}")
        print(f"   种群大小: {self.population_size}")
        print(f"   最大评估次数: {self.max_evaluations}")
        
        start_time = time.time()
        generation = 0
        
        while self.evaluations < self.max_evaluations:
            generation += 1
            
            # 生成新种群
            population = []
            fitness_values = []
            
            for i in range(self.population_size):
                # 采样
                z = np.random.multivariate_normal(np.zeros(self.dimension), self.C)
                x = self.mean + self.sigma * z
                
                # 转换为参数并评估
                params = self._vector_to_params(x)
                fitness = self._evaluate_individual(params)
                
                population.append(x)
                fitness_values.append(fitness)
                
                if self.evaluations >= self.max_evaluations:
                    break
            
            # 排序
            sorted_indices = np.argsort(fitness_values)
            population = [population[i] for i in sorted_indices]
            fitness_values = [fitness_values[i] for i in sorted_indices]
            
            # 更新分布参数
            if len(population) >= self.mu:
                # 选择最优个体
                selected = population[:self.mu]
                
                # 更新均值
                old_mean = self.mean.copy()
                self.mean = np.sum([self.weights[i] * selected[i] for i in range(self.mu)], axis=0)
                
                # 更新进化路径
                self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * np.linalg.solve(np.linalg.cholesky(self.C), self.mean - old_mean) / self.sigma
                
                hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * self.evaluations / self.population_size)) / self.chiN < 1.4 + 2 / (self.dimension + 1)
                
                self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * (self.mean - old_mean) / self.sigma
                
                # 更新协方差矩阵
                artmp = np.array([(selected[i] - old_mean) / self.sigma for i in range(self.mu)])
                self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * np.dot(artmp.T * self.weights, artmp)
                
                # 更新步长
                self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
            
            # 记录历史
            if len(fitness_values) > 0:
                self.history.append({
                    'generation': generation,
                    'evaluations': self.evaluations,
                    'best_fitness': min(fitness_values),
                    'mean_fitness': np.mean(fitness_values),
                    'sigma': self.sigma
                })
                
                if generation % 5 == 0 or self.evaluations >= self.max_evaluations:
                    print(f"   第 {generation} 代: 最佳适应度 = {min(fitness_values):.6f}, σ = {self.sigma:.4f}, 评估次数 = {self.evaluations}")
        
        end_time = time.time()
        
        print(f"✅ CMA-ES优化完成")
        print(f"   总耗时: {(end_time - start_time):.2f} 秒")
        print(f"   总评估次数: {self.evaluations}")
        print(f"   最佳适应度: {self.best_fitness:.6f}")
        
        if self.best_solution:
            print(f"   最佳解目标值:")
            obj = self.best_solution['objectives']
            print(f"     - 光学效率: {obj['f1_eff']:.4f}")
            print(f"     - 总成本: ${obj['f2_cost']:.2e}")
            print(f"     - 峰值热流: {obj['f3_flux']:.2f} kW/m²")
        
        return self.best_solution
    
    def save_results(self, output_path: Path):
        """保存优化结果"""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存最佳解
        if self.best_solution:
            with open(output_path / 'best_solution.json', 'w', encoding='utf-8') as f:
                json.dump(self.best_solution, f, indent=2, ensure_ascii=False)
        
        # 保存优化历史
        with open(output_path / 'optimization_history.json', 'w', encoding='utf-8') as f:
            json.dump({
                'upper_params': self.upper_params,
                'bounds': self.bounds,
                'max_evaluations': self.max_evaluations,
                'final_evaluations': self.evaluations,
                'best_fitness': self.best_fitness,
                'history': self.history
            }, f, indent=2, ensure_ascii=False)

def test_cmaes_optimizer():
    """测试CMA-ES优化器"""
    print("=== 测试CMA-ES下层优化器 ===")
    
    # 模拟上层参数
    upper_params = {
        'helio_az_spacing': 2.2,
        'helio_rad_spacing': 1.4
    }
    
    # 创建优化器
    optimizer = CMAESOptimizer(upper_params, max_evaluations=20)
    
    # 执行优化
    best_solution = optimizer.optimize()
    
    if best_solution:
        print("\n✅ CMA-ES优化器测试成功!")
        return best_solution
    else:
        print("\n❌ CMA-ES优化器测试失败!")
        return None

if __name__ == "__main__":
    test_cmaes_optimizer()