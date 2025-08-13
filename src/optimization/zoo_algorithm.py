#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合多目标鲸鱼优化-人工蜂群算法 (H-MOWOA-ABC)
用于定日镜场布局优化

基于以下算法的混合策略:
1. 多目标鲸鱼优化算法 (MOWOA)
2. 人工蜂群算法 (ABC)
3. 非支配排序和拥挤距离计算

作者: poboll
日期: 2025
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random
import copy
from dataclasses import dataclass

@dataclass
class HeliostatPosition:
    """定日镜位置数据结构"""
    x: float
    y: float
    id: int
    
@dataclass
class OptimizationResult:
    """优化结果数据结构"""
    positions: List[HeliostatPosition]
    objectives: Tuple[float, float, float]  # (年发电量, LCOE, 热通量均匀性)
    fitness: float
    generation: int

class H_MOWOA_ABC:
    """
    混合多目标鲸鱼优化-人工蜂群算法
    
    该算法结合了:
    - 鲸鱼优化算法的全局搜索能力
    - 人工蜂群算法的局部开发能力
    - 多目标优化的帕累托前沿生成
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 max_generations: int = 100,
                 field_bounds: Tuple[float, float, float, float] = (-500, 500, -500, 500),
                 num_heliostats: int = 1000,
                 tower_position: Tuple[float, float] = (0, 0),
                 heliostat_size: float = 115.7):
        """
        初始化H-MOWOA-ABC算法
        
        Args:
            population_size: 种群大小
            max_generations: 最大迭代次数
            field_bounds: 镜场边界 (xmin, xmax, ymin, ymax)
            num_heliostats: 定日镜数量
            tower_position: 集热塔位置
            heliostat_size: 单个定日镜面积 (m²)
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.field_bounds = field_bounds
        self.num_heliostats = num_heliostats
        self.tower_position = tower_position
        self.heliostat_size = heliostat_size
        
        # 算法参数
        self.a = 2.0  # 鲸鱼算法参数
        self.b = 1.0  # 螺旋形状参数
        self.limit = 10  # ABC算法限制参数
        
        # 存储结果
        self.population = []
        self.pareto_front = []
        self.history = []
        
    def initialize_population(self) -> List[List[HeliostatPosition]]:
        """
        初始化种群
        使用径向交错布局作为基础，添加随机扰动
        """
        population = []
        
        for i in range(self.population_size):
            individual = self._generate_radial_staggered_layout()
            # 添加随机扰动
            individual = self._add_random_perturbation(individual, perturbation_ratio=0.1)
            population.append(individual)
            
        return population
    
    def _generate_radial_staggered_layout(self) -> List[HeliostatPosition]:
        """
        生成径向交错布局作为基础布局
        """
        positions = []
        heliostat_id = 0
        
        # 计算定日镜间距（基于面积和安全距离）
        heliostat_width = np.sqrt(self.heliostat_size)
        spacing_x = heliostat_width * 1.5  # X方向间距
        spacing_y = heliostat_width * 1.2  # Y方向间距
        
        # 生成径向交错布局
        for radius in np.arange(50, 800, spacing_y):
            circumference = 2 * np.pi * radius
            num_in_ring = max(1, int(circumference / spacing_x))
            
            for i in range(num_in_ring):
                if heliostat_id >= self.num_heliostats:
                    break
                    
                angle = 2 * np.pi * i / num_in_ring
                # 交错排列
                if int(radius / spacing_y) % 2 == 1:
                    angle += np.pi / num_in_ring
                    
                x = self.tower_position[0] + radius * np.cos(angle)
                y = self.tower_position[1] + radius * np.sin(angle)
                
                # 检查边界
                if (self.field_bounds[0] <= x <= self.field_bounds[1] and 
                    self.field_bounds[2] <= y <= self.field_bounds[3]):
                    positions.append(HeliostatPosition(x, y, heliostat_id))
                    heliostat_id += 1
                    
            if heliostat_id >= self.num_heliostats:
                break
                
        return positions[:self.num_heliostats]
    
    def _add_random_perturbation(self, positions: List[HeliostatPosition], 
                                perturbation_ratio: float = 0.1) -> List[HeliostatPosition]:
        """
        为布局添加随机扰动
        """
        perturbed_positions = []
        max_perturbation = min(50, perturbation_ratio * 
                             (self.field_bounds[1] - self.field_bounds[0]))
        
        for pos in positions:
            dx = random.uniform(-max_perturbation, max_perturbation)
            dy = random.uniform(-max_perturbation, max_perturbation)
            
            new_x = np.clip(pos.x + dx, self.field_bounds[0], self.field_bounds[1])
            new_y = np.clip(pos.y + dy, self.field_bounds[2], self.field_bounds[3])
            
            perturbed_positions.append(HeliostatPosition(new_x, new_y, pos.id))
            
        return perturbed_positions
    
    def evaluate_objectives(self, positions: List[HeliostatPosition]) -> Tuple[float, float, float]:
        """
        评估三个目标函数:
        1. 年发电量 (最大化)
        2. 平准化电力成本 LCOE (最小化)
        3. 热通量均匀性 (最大化)
        
        注意: 这里使用简化的评估函数，实际应用中需要集成SolarPILOT
        """
        # 目标1: 年发电量估算 (基于光学效率)
        annual_energy = self._calculate_annual_energy(positions)
        
        # 目标2: LCOE估算
        lcoe = self._calculate_lcoe(positions, annual_energy)
        
        # 目标3: 热通量均匀性
        flux_uniformity = self._calculate_flux_uniformity(positions)
        
        return annual_energy, lcoe, flux_uniformity
    
    def _calculate_annual_energy(self, positions: List[HeliostatPosition]) -> float:
        """
        计算年发电量 (简化版本)
        实际应用中需要集成详细的光线追踪和热力学计算
        """
        total_optical_efficiency = 0.0
        
        for pos in positions:
            # 计算到塔的距离
            distance = np.sqrt((pos.x - self.tower_position[0])**2 + 
                             (pos.y - self.tower_position[1])**2)
            
            # 简化的光学效率模型
            cosine_efficiency = max(0.1, 1.0 - distance / 1000.0)  # 余弦效率
            atmospheric_attenuation = np.exp(-0.0001 * distance)    # 大气衰减
            shadowing_blocking = self._calculate_shadowing_blocking(pos, positions)
            
            optical_efficiency = cosine_efficiency * atmospheric_attenuation * shadowing_blocking
            total_optical_efficiency += optical_efficiency
        
        # 假设DNI = 2000 kWh/m²/year, 系统效率 = 0.4
        dni_annual = 2000  # kWh/m²/year
        system_efficiency = 0.4
        
        annual_energy = (total_optical_efficiency * self.heliostat_size * 
                        dni_annual * system_efficiency / 1000)  # MWh/year
        
        return annual_energy
    
    def _calculate_shadowing_blocking(self, target_pos: HeliostatPosition, 
                                    all_positions: List[HeliostatPosition]) -> float:
        """
        计算阴影和遮挡效率 (简化版本)
        """
        # 简化计算：基于最近邻距离
        distances = []
        for pos in all_positions:
            if pos.id != target_pos.id:
                dist = np.sqrt((pos.x - target_pos.x)**2 + (pos.y - target_pos.y)**2)
                distances.append(dist)
        
        if not distances:
            return 1.0
            
        min_distance = min(distances)
        heliostat_width = np.sqrt(self.heliostat_size)
        
        # 如果最近距离大于2倍定日镜宽度，认为无阴影
        if min_distance > 2 * heliostat_width:
            return 1.0
        else:
            return max(0.5, min_distance / (2 * heliostat_width))
    
    def _calculate_lcoe(self, positions: List[HeliostatPosition], annual_energy: float) -> float:
        """
        计算平准化电力成本 (简化版本)
        """
        # 简化的成本模型
        heliostat_cost = len(positions) * 150  # $/m² * 定日镜面积
        land_cost = self._calculate_land_area(positions) * 10  # $/m²
        tower_cost = 50000  # 固定成本
        
        total_capex = heliostat_cost + land_cost + tower_cost  # $
        
        # 运维成本 (年)
        opex_annual = total_capex * 0.02  # 2% of CAPEX
        
        # LCOE计算 (简化)
        project_lifetime = 25  # years
        discount_rate = 0.07
        
        if annual_energy <= 0:
            return float('inf')
            
        # 简化的LCOE计算
        lcoe = (total_capex + opex_annual * project_lifetime) / (annual_energy * project_lifetime)
        
        return lcoe  # $/MWh
    
    def _calculate_land_area(self, positions: List[HeliostatPosition]) -> float:
        """
        计算镜场占地面积
        """
        if not positions:
            return 0.0
            
        x_coords = [pos.x for pos in positions]
        y_coords = [pos.y for pos in positions]
        
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        return x_range * y_range  # m²
    
    def _calculate_flux_uniformity(self, positions: List[HeliostatPosition]) -> float:
        """
        计算接收器表面热通量均匀性 (简化版本)
        """
        # 简化模型：基于定日镜到塔的距离分布
        distances = []
        for pos in positions:
            dist = np.sqrt((pos.x - self.tower_position[0])**2 + 
                          (pos.y - self.tower_position[1])**2)
            distances.append(dist)
        
        if not distances:
            return 0.0
            
        # 计算距离的变异系数 (CV)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        if mean_dist == 0:
            return 0.0
            
        cv = std_dist / mean_dist
        # 均匀性 = 1 - CV (CV越小，均匀性越好)
        uniformity = max(0.0, 1.0 - cv)
        
        return uniformity
    
    def whale_optimization_update(self, individual: List[HeliostatPosition], 
                                best_individual: List[HeliostatPosition],
                                generation: int) -> List[HeliostatPosition]:
        """
        鲸鱼优化算法更新策略
        """
        # 更新参数a (从2线性递减到0)
        a = 2 * (1 - generation / self.max_generations)
        
        new_individual = []
        
        for i, pos in enumerate(individual):
            # 随机选择更新策略
            p = random.random()
            
            if p < 0.5:
                # 包围猎物策略
                if abs(a) < 1:
                    # 开发阶段
                    A = 2 * a * random.random() - a
                    C = 2 * random.random()
                    
                    D_x = abs(C * best_individual[i].x - pos.x)
                    D_y = abs(C * best_individual[i].y - pos.y)
                    
                    new_x = best_individual[i].x - A * D_x
                    new_y = best_individual[i].y - A * D_y
                else:
                    # 探索阶段
                    random_individual = random.choice(individual)
                    A = 2 * a * random.random() - a
                    C = 2 * random.random()
                    
                    D_x = abs(C * random_individual.x - pos.x)
                    D_y = abs(C * random_individual.y - pos.y)
                    
                    new_x = random_individual.x - A * D_x
                    new_y = random_individual.y - A * D_y
            else:
                # 螺旋更新策略
                l = random.uniform(-1, 1)
                
                D_x = abs(best_individual[i].x - pos.x)
                D_y = abs(best_individual[i].y - pos.y)
                
                new_x = D_x * np.exp(self.b * l) * np.cos(2 * np.pi * l) + best_individual[i].x
                new_y = D_y * np.exp(self.b * l) * np.cos(2 * np.pi * l) + best_individual[i].y
            
            # 边界检查
            new_x = np.clip(new_x, self.field_bounds[0], self.field_bounds[1])
            new_y = np.clip(new_y, self.field_bounds[2], self.field_bounds[3])
            
            new_individual.append(HeliostatPosition(new_x, new_y, pos.id))
        
        return new_individual
    
    def abc_local_search(self, individual: List[HeliostatPosition]) -> List[HeliostatPosition]:
        """
        人工蜂群算法局部搜索
        """
        new_individual = copy.deepcopy(individual)
        
        # 随机选择一定比例的定日镜进行局部搜索
        num_to_modify = max(1, int(0.1 * len(individual)))
        indices_to_modify = random.sample(range(len(individual)), num_to_modify)
        
        for idx in indices_to_modify:
            # 在邻域内搜索
            search_radius = 20.0  # 搜索半径
            
            dx = random.uniform(-search_radius, search_radius)
            dy = random.uniform(-search_radius, search_radius)
            
            new_x = np.clip(individual[idx].x + dx, 
                           self.field_bounds[0], self.field_bounds[1])
            new_y = np.clip(individual[idx].y + dy, 
                           self.field_bounds[2], self.field_bounds[3])
            
            new_individual[idx] = HeliostatPosition(new_x, new_y, individual[idx].id)
        
        return new_individual
    
    def non_dominated_sort(self, population_objectives: List[Tuple[float, float, float]]) -> List[List[int]]:
        """
        非支配排序
        """
        n = len(population_objectives)
        domination_count = [0] * n  # 被支配次数
        dominated_solutions = [[] for _ in range(n)]  # 支配的解
        fronts = [[]]
        
        # 计算支配关系
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(population_objectives[i], population_objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(population_objectives[j], population_objectives[i]):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        # 构建后续前沿
        front_idx = 0
        while front_idx < len(fronts) and len(fronts[front_idx]) > 0:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            if next_front:
                fronts.append(next_front)
            front_idx += 1
        
        # 移除空的前沿
        fronts = [front for front in fronts if len(front) > 0]
        
        return fronts
    
    def _dominates(self, obj1: Tuple[float, float, float], 
                  obj2: Tuple[float, float, float]) -> bool:
        """
        判断obj1是否支配obj2
        目标1 (年发电量): 最大化
        目标2 (LCOE): 最小化  
        目标3 (热通量均匀性): 最大化
        """
        # 转换为最小化问题进行比较
        obj1_min = (-obj1[0], obj1[1], -obj1[2])  # 年发电量和均匀性取负号
        obj2_min = (-obj2[0], obj2[1], -obj2[2])
        
        # 检查是否至少在一个目标上更好，且在所有目标上不差
        better_in_at_least_one = False
        for i in range(3):
            if obj1_min[i] > obj2_min[i]:  # obj1在目标i上更差
                return False
            elif obj1_min[i] < obj2_min[i]:  # obj1在目标i上更好
                better_in_at_least_one = True
        
        return better_in_at_least_one
    
    def calculate_crowding_distance(self, front: List[int], 
                                  population_objectives: List[Tuple[float, float, float]]) -> List[float]:
        """
        计算拥挤距离
        """
        if len(front) <= 2:
            return [float('inf')] * len(front)
        
        distances = [0.0] * len(front)
        
        # 对每个目标计算拥挤距离
        for obj_idx in range(3):
            # 按目标值排序
            front_sorted = sorted(front, key=lambda x: population_objectives[x][obj_idx])
            
            # 边界点设为无穷大
            idx_min = front.index(front_sorted[0])
            idx_max = front.index(front_sorted[-1])
            distances[idx_min] = float('inf')
            distances[idx_max] = float('inf')
            
            # 计算目标值范围
            obj_min = population_objectives[front_sorted[0]][obj_idx]
            obj_max = population_objectives[front_sorted[-1]][obj_idx]
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            # 计算中间点的拥挤距离
            for i in range(1, len(front_sorted) - 1):
                idx = front.index(front_sorted[i])
                if distances[idx] != float('inf'):
                    prev_obj = population_objectives[front_sorted[i-1]][obj_idx]
                    next_obj = population_objectives[front_sorted[i+1]][obj_idx]
                    distances[idx] += (next_obj - prev_obj) / obj_range
        
        return distances
    
    def optimize(self) -> Dict[str, Any]:
        """
        执行H-MOWOA-ABC优化算法
        
        Returns:
            包含优化结果的字典
        """
        print("开始H-MOWOA-ABC优化算法...")
        print(f"种群大小: {self.population_size}")
        print(f"最大迭代次数: {self.max_generations}")
        print(f"定日镜数量: {self.num_heliostats}")
        print("-" * 50)
        
        # 初始化种群
        population = self.initialize_population()
        
        # 评估初始种群
        population_objectives = []
        for individual in population:
            try:
                objectives = self.evaluate_objectives(individual)
                population_objectives.append(objectives)
            except Exception as e:
                print(f"警告: 个体评估失败 - {str(e)}")
                # 使用默认值
                population_objectives.append((0.0, 1000.0, 0.0))
        
        best_individual = None
        best_fitness = float('-inf')
        
        # 主优化循环
        for generation in range(self.max_generations):
            print(f"第 {generation + 1}/{self.max_generations} 代")
            
            # 非支配排序
            fronts = self.non_dominated_sort(population_objectives)
            
            # 更新最佳个体 (第一前沿的第一个)
            if fronts and fronts[0]:
                first_front_idx = fronts[0][0]
                current_fitness = sum(population_objectives[first_front_idx])
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_individual = copy.deepcopy(population[first_front_idx])
            
            # 生成新种群
            new_population = []
            new_objectives = []
            
            for i, individual in enumerate(population):
                # 鲸鱼优化算法更新
                if best_individual:
                    new_individual = self.whale_optimization_update(
                        individual, best_individual, generation)
                else:
                    new_individual = individual
                
                # ABC局部搜索 (概率性应用)
                if random.random() < 0.3:  # 30%概率进行局部搜索
                    new_individual = self.abc_local_search(new_individual)
                
                # 评估新个体
                new_objectives_val = self.evaluate_objectives(new_individual)
                
                # 选择更好的个体
                if self._dominates(new_objectives_val, population_objectives[i]):
                    new_population.append(new_individual)
                    new_objectives.append(new_objectives_val)
                else:
                    new_population.append(individual)
                    new_objectives.append(population_objectives[i])
            
            population = new_population
            population_objectives = new_objectives
            
            # 记录历史
            generation_stats = {
                'generation': generation + 1,
                'best_fitness': best_fitness,
                'pareto_front_size': len(fronts[0]) if fronts else 0
            }
            self.history.append(generation_stats)
            
            if (generation + 1) % 10 == 0:
                print(f"  最佳适应度: {best_fitness:.2f}")
                print(f"  帕累托前沿大小: {len(fronts[0]) if fronts else 0}")
        
        # 提取最终帕累托前沿
        final_fronts = self.non_dominated_sort(population_objectives)
        if final_fronts and final_fronts[0]:
            self.pareto_front = []
            for idx in final_fronts[0]:
                result = OptimizationResult(
                    positions=population[idx],
                    objectives=population_objectives[idx],
                    fitness=sum(population_objectives[idx]),
                    generation=self.max_generations
                )
                self.pareto_front.append(result)
        
        print("\n优化完成!")
        print(f"最终帕累托前沿包含 {len(self.pareto_front)} 个解")
        
        return {
            'pareto_front': self.pareto_front,
            'best_individual': best_individual,
            'history': self.history,
            'final_population': population,
            'final_objectives': population_objectives
        }
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "results"):
        """
        保存优化结果
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存帕累托前沿
        pareto_data = []
        for i, result in enumerate(results['pareto_front']):
            pareto_data.append({
                'solution_id': i + 1,
                'annual_energy_MWh': result.objectives[0],
                'lcoe_USD_per_MWh': result.objectives[1],
                'flux_uniformity': result.objectives[2],
                'fitness': result.fitness
            })
        
        pareto_df = pd.DataFrame(pareto_data)
        pareto_df.to_csv(f"{output_dir}/pareto_front.csv", index=False)
        
        # 保存每个解的布局
        layouts_dir = f"{output_dir}/optimal_layouts"
        os.makedirs(layouts_dir, exist_ok=True)
        
        for i, result in enumerate(results['pareto_front']):
            layout_data = []
            for pos in result.positions:
                layout_data.append({
                    'heliostat_id': pos.id,
                    'x_coordinate': pos.x,
                    'y_coordinate': pos.y
                })
            
            layout_df = pd.DataFrame(layout_data)
            layout_df.to_csv(f"{layouts_dir}/solution_{i+1:03d}.csv", index=False)
        
        # 保存优化历史
        history_df = pd.DataFrame(results['history'])
        history_df.to_csv(f"{output_dir}/optimization_history.csv", index=False)
        
        print(f"结果已保存到 {output_dir} 目录")
    
    def plot_results(self, results: Dict[str, Any]):
        """
        可视化优化结果
        """
        # 绘制帕累托前沿
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 提取目标值
        annual_energy = [r.objectives[0] for r in results['pareto_front']]
        lcoe = [r.objectives[1] for r in results['pareto_front']]
        flux_uniformity = [r.objectives[2] for r in results['pareto_front']]
        
        # 年发电量 vs LCOE
        axes[0, 0].scatter(annual_energy, lcoe, c='red', alpha=0.7)
        axes[0, 0].set_xlabel('年发电量 (MWh)')
        axes[0, 0].set_ylabel('LCOE ($/MWh)')
        axes[0, 0].set_title('年发电量 vs LCOE')
        axes[0, 0].grid(True)
        
        # 年发电量 vs 热通量均匀性
        axes[0, 1].scatter(annual_energy, flux_uniformity, c='blue', alpha=0.7)
        axes[0, 1].set_xlabel('年发电量 (MWh)')
        axes[0, 1].set_ylabel('热通量均匀性')
        axes[0, 1].set_title('年发电量 vs 热通量均匀性')
        axes[0, 1].grid(True)
        
        # LCOE vs 热通量均匀性
        axes[1, 0].scatter(lcoe, flux_uniformity, c='green', alpha=0.7)
        axes[1, 0].set_xlabel('LCOE ($/MWh)')
        axes[1, 0].set_ylabel('热通量均匀性')
        axes[1, 0].set_title('LCOE vs 热通量均匀性')
        axes[1, 0].grid(True)
        
        # 优化历史
        history_df = pd.DataFrame(results['history'])
        axes[1, 1].plot(history_df['generation'], history_df['best_fitness'], 'b-', linewidth=2)
        axes[1, 1].set_xlabel('迭代次数')
        axes[1, 1].set_ylabel('最佳适应度')
        axes[1, 1].set_title('优化收敛历史')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 绘制最佳布局
        if results['pareto_front']:
            best_solution = max(results['pareto_front'], key=lambda x: x.fitness)
            
            plt.figure(figsize=(12, 10))
            x_coords = [pos.x for pos in best_solution.positions]
            y_coords = [pos.y for pos in best_solution.positions]
            
            plt.scatter(x_coords, y_coords, c='blue', alpha=0.6, s=20)
            plt.scatter(self.tower_position[0], self.tower_position[1], 
                       c='red', s=200, marker='^', label='集热塔')
            
            plt.xlabel('X 坐标 (m)')
            plt.ylabel('Y 坐标 (m)')
            plt.title(f'最佳定日镜场布局\n年发电量: {best_solution.objectives[0]:.1f} MWh, '
                     f'LCOE: {best_solution.objectives[1]:.1f} $/MWh')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            
            plt.savefig('results/best_layout.png', dpi=300, bbox_inches='tight')
            plt.show()


if __name__ == "__main__":
    # 示例使用
    print("H-MOWOA-ABC 定日镜场优化算法")
    print("=" * 50)
    
    # 创建优化器实例
    optimizer = H_MOWOA_ABC(
        population_size=30,
        max_generations=50,
        field_bounds=(-800, 800, -800, 800),
        num_heliostats=500,
        tower_position=(0, 0),
        heliostat_size=115.7
    )
    
    # 执行优化
    results = optimizer.optimize()
    
    # 保存结果
    optimizer.save_results(results, "results")
    
    # 可视化结果
    optimizer.plot_results(results)
    
    print("\n优化完成！结果已保存到 results 目录。")