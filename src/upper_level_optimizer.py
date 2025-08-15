# file: src/upper_level_optimizer.py

import random
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from pysam_interface import evaluate_fitness

class Individual:
    """个体类，表示一个解决方案"""
    def __init__(self, params: Dict[str, float]):
        self.params = params.copy()
        self.objectives = None  # [f1, f2, f3] - 目标函数值
        self.fitness = None     # 适应度值
        self.rank = None        # 非支配排序等级
        self.crowding_distance = 0.0  # 拥挤距离
        self.dominated_count = 0      # 被支配个体数量
        self.dominating_set = []      # 支配的个体集合
        
    def dominates(self, other) -> bool:
        """判断当前个体是否支配另一个个体"""
        if self.objectives is None or other.objectives is None:
            return False
            
        # 至少在一个目标上更好，且在所有目标上不差
        better_in_any = False
        for i in range(len(self.objectives)):
            if self.objectives[i] > other.objectives[i]:  # 假设最小化问题
                return False
            elif self.objectives[i] < other.objectives[i]:
                better_in_any = True
                
        return better_in_any
        
    def copy(self):
        """创建个体的深拷贝"""
        new_individual = Individual(self.params)
        new_individual.objectives = self.objectives.copy() if self.objectives else None
        new_individual.fitness = self.fitness
        return new_individual

class HeliostatOptimizer:
    """基于H-MOWOA-ABC的定日镜场布局多目标优化器"""
    
    def __init__(self, population_size=50, max_generations=100, weather_file_path=None):
        self.population_size = population_size
        self.max_generations = max_generations
        self.weather_file_path = weather_file_path
        
        # 决策变量边界 (基于敦煌项目经验)
        self.bounds = {
            'helio_az_spacing': [1.8, 2.8],    # 方位角间距倍数
            'helio_rad_spacing': [1.2, 2.2],   # 径向间距倍数
            # 可以添加更多优化变量
            # 'helio_width': [10.0, 15.0],       # 定日镜宽度
            # 'helio_height': [10.0, 15.0],      # 定日镜高度
        }
        
        self.population = []  # 当前种群
        self.archive = []     # 外部档案 (帕累托前沿)
        self.generation = 0
        self.evaluation_count = 0
        
        # H-MOWOA-ABC算法参数
        self.whale_ratio = 0.6      # 鲸鱼算法个体比例
        self.abc_ratio = 0.4        # 蜜蜂算法个体比例
        self.limit = 10             # ABC算法中的limit参数
        self.a_max = 2.0            # WOA算法中的a参数最大值
        
        # Kent映射参数
        self.kent_mu = 0.7          # Kent映射参数
        
        # 历史记录
        self.history = {
            'generations': [],
            'best_objectives': [],
            'population_diversity': [],
            'evaluation_times': []
        }
        
    def _kent_map(self, x: float) -> float:
        """Kent混沌映射"""
        if x < self.kent_mu:
            return x / self.kent_mu
        else:
            return (1 - x) / (1 - self.kent_mu)
            
    def _initialize_population(self):
        """使用Kent映射初始化种群"""
        print(f"🔄 初始化种群 (大小: {self.population_size})")
        self.population = []
        
        for i in range(self.population_size):
            # 使用Kent映射生成初始参数
            params = {}
            x = random.random()  # 初始随机值
            
            for param_name, (min_val, max_val) in self.bounds.items():
                x = self._kent_map(x)  # 应用Kent映射
                params[param_name] = min_val + x * (max_val - min_val)
                
            individual = Individual(params)
            self.population.append(individual)
            
        print(f"✅ 种群初始化完成")
        
    def _evaluate_population(self):
        """评估种群中所有个体的适应度"""
        print(f"🔄 评估第 {self.generation + 1} 代种群适应度")
        
        unevaluated_count = 0
        for individual in self.population:
            if individual.objectives is None:
                unevaluated_count += 1
                
        print(f"   需要评估的个体数量: {unevaluated_count}")
        
        for i, individual in enumerate(self.population):
            if individual.objectives is None:  # 只评估新个体
                start_time = time.time()
                
                # 调用PySAM仿真
                results = evaluate_fitness(individual.params, self.weather_file_path)
                
                eval_time = time.time() - start_time
                self.history['evaluation_times'].append(eval_time)
                
                if results:
                    # 转换为最小化问题的目标函数值
                    individual.objectives = [
                        -results['f1_eff'],      # f1: 最大化光学效率 -> 最小化负效率
                        results['f2_cost'],      # f2: 最小化总成本
                        results['f3_flux']       # f3: 最小化峰值热流密度
                    ]
                    self.evaluation_count += 1
                    print(f"   个体 {i+1}/{self.population_size} 评估完成 (耗时: {eval_time:.2f}s)")
                else:
                    # 仿真失败，设置惩罚值
                    individual.objectives = [float('inf'), float('inf'), float('inf')]
                    print(f"   个体 {i+1}/{self.population_size} 评估失败")
                    
        print(f"✅ 种群评估完成，累计评估次数: {self.evaluation_count}")
        
    def _fast_non_dominated_sort(self) -> List[List[Individual]]:
        """快速非支配排序 (NSGA-II)"""
        fronts = [[]]
        
        for individual in self.population:
            individual.dominated_count = 0
            individual.dominating_set = []
            
            for other in self.population:
                if individual.dominates(other):
                    individual.dominating_set.append(other)
                elif other.dominates(individual):
                    individual.dominated_count += 1
                    
            if individual.dominated_count == 0:
                individual.rank = 0
                fronts[0].append(individual)
                
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for individual in fronts[i]:
                for dominated in individual.dominating_set:
                    dominated.dominated_count -= 1
                    if dominated.dominated_count == 0:
                        dominated.rank = i + 1
                        next_front.append(dominated)
            i += 1
            fronts.append(next_front)
            
        return fronts[:-1]  # 移除最后的空前沿
        
    def _calculate_crowding_distance(self, front: List[Individual]):
        """计算拥挤距离"""
        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = float('inf')
            return
            
        # 初始化拥挤距离
        for individual in front:
            individual.crowding_distance = 0.0
            
        n_objectives = len(front[0].objectives)
        
        for obj_idx in range(n_objectives):
            # 按第obj_idx个目标排序
            front.sort(key=lambda x: x.objectives[obj_idx])
            
            # 边界个体设置为无穷大
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # 计算目标函数范围
            obj_range = front[-1].objectives[obj_idx] - front[0].objectives[obj_idx]
            if obj_range == 0:
                continue
                
            # 计算中间个体的拥挤距离
            for i in range(1, len(front) - 1):
                distance = (front[i+1].objectives[obj_idx] - front[i-1].objectives[obj_idx]) / obj_range
                front[i].crowding_distance += distance
                
    def _whale_optimization_update(self, individual: Individual, best_individual: Individual) -> Individual:
        """鲸鱼优化算法更新策略"""
        new_params = {}
        
        # 计算a参数 (随代数线性递减)
        a = self.a_max * (1 - self.generation / self.max_generations)
        
        for param_name, (min_val, max_val) in self.bounds.items():
            r1, r2 = random.random(), random.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            
            if abs(A) < 1:
                # 包围猎物 (exploitation)
                D = abs(C * best_individual.params[param_name] - individual.params[param_name])
                new_val = best_individual.params[param_name] - A * D
            else:
                # 搜索猎物 (exploration)
                rand_individual = random.choice(self.population)
                D = abs(C * rand_individual.params[param_name] - individual.params[param_name])
                new_val = rand_individual.params[param_name] - A * D
                
            # 边界处理
            new_val = max(min_val, min(max_val, new_val))
            new_params[param_name] = new_val
            
        return Individual(new_params)
        
    def _abc_update(self, individual: Individual) -> Individual:
        """人工蜂群算法更新策略"""
        new_params = individual.params.copy()
        
        # 随机选择一个参数进行更新
        param_name = random.choice(list(self.bounds.keys()))
        min_val, max_val = self.bounds[param_name]
        
        # 检查是否有足够的个体进行选择
        available_partners = [ind for ind in self.population if ind != individual]
        if not available_partners:
            # 如果没有其他个体，进行随机扰动
            perturbation = random.uniform(-0.1, 0.1) * (max_val - min_val)
            new_val = individual.params[param_name] + perturbation
        else:
            # 选择一个不同的个体
            partner = random.choice(available_partners)
            
            # ABC更新公式
            phi = random.uniform(-1, 1)
            new_val = individual.params[param_name] + phi * (individual.params[param_name] - partner.params[param_name])
        
        # 边界处理
        new_val = max(min_val, min(max_val, new_val))
        new_params[param_name] = new_val
        
        return Individual(new_params)
        
    def _update_population(self):
        """更新种群 (H-MOWOA-ABC核心逻辑)"""
        print(f"🔄 更新第 {self.generation + 1} 代种群")
        
        # 非支配排序
        fronts = self._fast_non_dominated_sort()
        
        # 计算拥挤距离
        for front in fronts:
            self._calculate_crowding_distance(front)
            
        # 选择最佳个体 (第一前沿中拥挤距离最大的)
        if len(fronts) > 0 and len(fronts[0]) > 0:
            best_individual = max(fronts[0], key=lambda x: x.crowding_distance if x.crowding_distance != float('inf') else 0)
        else:
            best_individual = random.choice(self.population)
            
        # 生成新种群
        new_population = []
        
        # 保留精英 (第一前沿)
        if len(fronts) > 0:
            elite_size = min(len(fronts[0]), self.population_size // 4)
            elite = sorted(fronts[0], key=lambda x: x.crowding_distance, reverse=True)[:elite_size]
            new_population.extend([ind.copy() for ind in elite])
            
        # 生成新个体
        while len(new_population) < self.population_size:
            if random.random() < self.whale_ratio:
                # 使用鲸鱼优化算法
                parent = random.choice(self.population)
                offspring = self._whale_optimization_update(parent, best_individual)
            else:
                # 使用人工蜂群算法
                parent = random.choice(self.population)
                offspring = self._abc_update(parent)
                
            new_population.append(offspring)
            
        self.population = new_population[:self.population_size]
        print(f"✅ 种群更新完成")
        
    def _update_archive(self):
        """更新外部档案 (帕累托前沿)"""
        # 合并当前种群和档案
        combined = self.archive + [ind for ind in self.population if ind.objectives is not None]
        
        # 非支配排序
        temp_population = self.population
        self.population = combined
        fronts = self._fast_non_dominated_sort()
        self.population = temp_population
        
        # 更新档案为第一前沿
        if len(fronts) > 0:
            self.archive = [ind.copy() for ind in fronts[0]]
            
    def _calculate_diversity(self) -> float:
        """计算种群多样性"""
        if len(self.population) < 2:
            return 0.0
            
        total_distance = 0.0
        count = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                if self.population[i].objectives and self.population[j].objectives:
                    distance = np.linalg.norm(
                        np.array(self.population[i].objectives) - np.array(self.population[j].objectives)
                    )
                    total_distance += distance
                    count += 1
                    
        return total_distance / count if count > 0 else 0.0
        
    def _save_generation_info(self):
        """保存当前代的信息"""
        # 找到当前代最佳目标值
        valid_individuals = [ind for ind in self.population if ind.objectives is not None]
        if valid_individuals:
            best_objectives = []
            for obj_idx in range(3):  # 三个目标函数
                best_val = min(ind.objectives[obj_idx] for ind in valid_individuals)
                best_objectives.append(best_val)
        else:
            best_objectives = [float('inf')] * 3
            
        diversity = self._calculate_diversity()
        
        self.history['generations'].append(self.generation)
        self.history['best_objectives'].append(best_objectives)
        self.history['population_diversity'].append(diversity)
        
        print(f"   第 {self.generation + 1} 代最佳目标值: {[f'{obj:.4f}' for obj in best_objectives]}")
        print(f"   种群多样性: {diversity:.4f}")
        
    def run(self) -> List[Individual]:
        """运行优化算法"""
        print("=== 开始运行H-MOWOA-ABC多目标优化算法 ===")
        print(f"种群大小: {self.population_size}")
        print(f"最大代数: {self.max_generations}")
        print(f"决策变量: {list(self.bounds.keys())}")
        print(f"变量边界: {self.bounds}")
        
        start_time = time.time()
        
        # 初始化种群
        self._initialize_population()
        
        # 进化循环
        for gen in range(self.max_generations):
            self.generation = gen
            print(f"\n--- 第 {gen + 1}/{self.max_generations} 代优化 ---")
            
            # 评估种群
            self._evaluate_population()
            
            # 更新外部档案
            self._update_archive()
            
            # 保存当前代信息
            self._save_generation_info()
            
            # 更新种群 (除了最后一代)
            if gen < self.max_generations - 1:
                self._update_population()
                
        total_time = time.time() - start_time
        
        print(f"\n=== 优化完成 ===")
        print(f"总耗时: {total_time / 60:.2f} 分钟")
        print(f"总评估次数: {self.evaluation_count}")
        print(f"平均每次评估耗时: {np.mean(self.history['evaluation_times']):.2f} 秒")
        print(f"最终帕累托前沿大小: {len(self.archive)}")
        
        return self.archive
        
    def save_results(self, output_dir: Path):
        """保存优化结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存帕累托前沿
        pareto_data = []
        for i, individual in enumerate(self.archive):
            data = {
                'solution_id': i,
                'f1_efficiency': -individual.objectives[0],  # 转换回正值
                'f2_cost': individual.objectives[1],
                'f3_flux': individual.objectives[2],
                **individual.params
            }
            pareto_data.append(data)
            
        with open(output_dir / 'pareto_front.json', 'w', encoding='utf-8') as f:
            json.dump(pareto_data, f, indent=2, ensure_ascii=False)
            
        # 保存优化历史
        with open(output_dir / 'optimization_history.json', 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
            
        print(f"✅ 结果已保存到: {output_dir}")

def test_optimizer():
    """测试优化器"""
    print("=== 测试H-MOWOA-ABC优化器 ===")
    
    # 使用小规模参数进行快速测试
    optimizer = HeliostatOptimizer(
        population_size=4,
        max_generations=2
    )
    
    # 运行优化
    pareto_front = optimizer.run()
    
    print(f"\n测试完成，获得 {len(pareto_front)} 个帕累托最优解")
    
    return pareto_front

if __name__ == "__main__":
    test_optimizer()