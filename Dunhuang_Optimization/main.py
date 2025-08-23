# file: main.py

from src.upper_level_optimizer import HeliostatOptimizer
import time
import pandas as pd

if __name__ == "__main__":
    print("=== 开始运行敦煌定日镜场布局多目标优化 ===")
    start_time = time.time()

    # 使用较小的参数进行快速测试
    optimizer = HeliostatOptimizer(population_size=10, max_generations=5) 
    
    # 运行优化
    final_solutions = optimizer.run()

    end_time = time.time()
    print(f"\n=== 优化结束。总耗时: {(end_time - start_time) / 60:.2f} 分钟 ===")

    # 保存结果到CSV
    if final_solutions:
        df = pd.DataFrame(final_solutions)
        df.to_csv("results/pareto_front.csv", index=False)
        print("✅ 最终帕累托前沿已保存至 'results/pareto_front.csv'")
    else:
        print("⚠️ 未获得有效解。")