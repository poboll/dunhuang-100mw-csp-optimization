#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
敦煌100MW熔盐塔式光热电站定日镜场布局可视化
基于优化结果生成定日镜场布局图
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_heliostat_layout(az_spacing, rad_spacing, tower_pos=(0, 0), max_radius=800, max_heliostats=12500, mirror_width=11.0, safety_gap=1.3):
    """
    创建定日镜场布局，增加数量上限、径向交错和可变密度。

    Args:
        az_spacing: 基础方位角中心距 (m)
        rad_spacing: 基础径向中心距 (m)
        tower_pos: 塔位置 (x, y)
        max_radius: 镜场最大半径 (m)
        max_heliostats: 最大定日镜数量
        mirror_width: 单面定日镜宽度 (m)
        safety_gap: 镜子之间额外安全系数 (>=1.0)，1.3 表示留出 30% 间隙

    Returns:
        heliostat_positions: 定日镜中心点坐标列表 [(x, y), ...]
    """
    positions = []
    total_heliostats = 0
    is_odd_ring = True

    # 距塔最小安全半径（塔周施工区），默认 60m
    current_radius = 60.0

    while current_radius < max_radius and total_heliostats < max_heliostats:
        # 考虑“近密远疏”的动态放大系数
        scale_factor = 1 + 0.5 * (current_radius / max_radius)

        # 计算当前环的实际中心距，确保 ≥ 镜子宽度 * safety_gap
        dynamic_rad_spacing = max(rad_spacing * scale_factor, mirror_width * safety_gap)
        dynamic_az_spacing = max(az_spacing * scale_factor, mirror_width * safety_gap)

        circumference = 2 * np.pi * current_radius
        n_heliostats_in_ring = int(circumference / dynamic_az_spacing)

        if n_heliostats_in_ring <= 0:
            current_radius += dynamic_rad_spacing
            continue

        # 若超出总数上限则截断
        if total_heliostats + n_heliostats_in_ring > max_heliostats:
            n_heliostats_in_ring = max_heliostats - total_heliostats

        angle_spacing = 2 * np.pi / n_heliostats_in_ring
        angle_offset = angle_spacing / 2 if is_odd_ring else 0.0

        for i in range(n_heliostats_in_ring):
            angle = i * angle_spacing + angle_offset
            x = tower_pos[0] + current_radius * np.cos(angle)
            y = tower_pos[1] + current_radius * np.sin(angle)
            positions.append((x, y))

        total_heliostats += n_heliostats_in_ring
        current_radius += dynamic_rad_spacing
        is_odd_ring = not is_odd_ring

    return positions

def visualize_best_layout():
    """
    可视化最佳布局方案
    """
    # 读取帕累托前沿数据
    results_dir = Path("/Users/Apple/Downloads/官方主题/results/optimization")
    pareto_file = results_dir / "pareto_front.csv"
    
    if not pareto_file.exists():
        print("❌ 未找到优化结果文件")
        return
    
    # 读取数据
    df = pd.read_csv(pareto_file)
    print(f"📊 读取到 {len(df)} 个帕累托最优解")
    
    # 选择最佳效率方案
    best_efficiency_idx = df['optical_efficiency'].idxmax()
    best_solution = df.iloc[best_efficiency_idx]
    
    print(f"\n🎯 最佳效率方案:")
    print(f"   光学效率: {best_solution['optical_efficiency']:.4f}")
    print(f"   总成本: ${best_solution['total_cost_usd']:.0f}")
    print(f"   峰值热流: {best_solution['peak_flux_kw_m2']:.2f} kW/m²")
    print(f"   方位角间距: {best_solution['helio_az_spacing']:.2f} m")
    print(f"   径向间距: {best_solution['helio_rad_spacing']:.2f} m")
    
    # 生成布局
    positions = create_heliostat_layout(
        az_spacing=best_solution['helio_az_spacing'],
        rad_spacing=best_solution['helio_rad_spacing'],
        tower_pos=(0, 0),
        max_radius=800,
        max_heliostats=12500
    )
    positions = apply_elliptical_mask(positions, orientation_deg=90)
    print(f"\n🔢 裁剪后定日镜数量: {len(positions)}")
    # 创建可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：完整布局
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]

    ax1.scatter(x_coords, y_coords, s=0.1, alpha=0.8, c='#003366', label='定日镜') # 调整点尺寸
    ax1.scatter(0, 0, s=100, c='red', marker='s', label='中央塔', zorder=5)

    # 添加同心圆参考线
    for r in [200, 400, 600, 800]:
        circle = plt.Circle((0, 0), r, fill=False, linestyle='--', alpha=0.3, color='gray')
        ax1.add_patch(circle)
        ax1.text(r*0.7, r*0.7, f'{r}m', fontsize=8, alpha=0.7)
    
    ax1.set_xlim(-900, 900)
    ax1.set_ylim(-900, 900)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('东西方向 (m)')
    ax1.set_ylabel('南北方向 (m)')
    ax1.set_title(f'敦煌100MW熔盐塔式光热电站定日镜场布局\n(方位角间距: {best_solution["helio_az_spacing"]:.2f}m, 径向间距: {best_solution["helio_rad_spacing"]:.2f}m)')
    ax1.legend()
    
    # 右图：局部放大
    ax2.scatter(x_coords, y_coords, s=2, alpha=0.9, c='#003366', label='定日镜') # 调整点尺寸
    ax2.scatter(0, 0, s=200, c='red', marker='s', label='中央塔', zorder=5)

    ax2.set_xlim(-300, 300)
    ax2.set_ylim(-300, 300)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('东西方向 (m)')
    ax2.set_ylabel('南北方向 (m)')
    ax2.set_title('中心区域放大图')
    ax2.legend()
    
    # 添加性能指标文本
    info_text = f"""性能指标:
光学效率: {best_solution['optical_efficiency']:.3f}
建设成本: ${best_solution['total_cost_usd']/1e8:.2f}亿
峰值热流: {best_solution['peak_flux_kw_m2']:.1f} kW/m²
定日镜数量: {len(positions):,}
镜场半径: 800m"""
    
    ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    output_file = results_dir / "heliostat_field_layout.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n💾 布局图已保存: {output_file}")
    
    plt.show()
    
    return positions, best_solution

def compare_layouts():
    """
    比较不同布局方案
    """
    results_dir = Path("/Users/Apple/Downloads/官方主题/results/optimization")
    pareto_file = results_dir / "pareto_front.csv"
    
    if not pareto_file.exists():
        print("❌ 未找到优化结果文件")
        return
    
    df = pd.read_csv(pareto_file)
    
    # 选择几个代表性方案
    best_efficiency = df.loc[df['optical_efficiency'].idxmax()]
    best_cost = df.loc[df['total_cost_usd'].idxmin()]
    best_flux = df.loc[df['peak_flux_kw_m2'].idxmin()]
    
    solutions = {
        '最高效率方案': best_efficiency,
        '最低成本方案': best_cost,
        '最低热流方案': best_flux
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (name, solution) in enumerate(solutions.items()):
        positions = create_heliostat_layout(
            az_spacing=solution['helio_az_spacing'],
            rad_spacing=solution['helio_rad_spacing'],
            max_radius=800,
            max_heliostats=12500
        )
        positions = apply_elliptical_mask(positions, orientation_deg=90)
        
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        axes[i].scatter(x_coords, y_coords, s=0.1, alpha=0.7, c='#003366') # 减小点尺寸
        axes[i].scatter(0, 0, s=50, c='red', marker='s', zorder=5)
        
        axes[i].set_xlim(-700, 700)
        axes[i].set_ylim(-700, 700)
        axes[i].set_aspect('equal')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f'{name}\n效率:{solution["optical_efficiency"]:.3f}, 成本:${solution["total_cost_usd"]/1e8:.2f}亿\n间距:({solution["helio_az_spacing"]:.2f}, {solution["helio_rad_spacing"]:.2f})m')
        axes[i].set_xlabel('东西方向 (m)')
        if i == 0:
            axes[i].set_ylabel('南北方向 (m)')
    
    plt.tight_layout()
    
    # 保存对比图
    output_file = results_dir / "layout_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n💾 对比图已保存: {output_file}")
    
    plt.show()

def apply_elliptical_mask(positions, a_radius=800, b_radius=700, orientation_deg=90):
    """根据任意旋转椭圆 (x'/a)^2+(y'/b)^2≤1 过滤定日镜位置
    
    参数
    -----
    positions : List[Tuple[float,float]]
    待筛选的 (x, y) 坐标列表，单位 m。
    a_radius : float, 默认 800
    椭圆长半轴长度 (m)。
    b_radius : float, 默认 700
    椭圆短半轴长度 (m)。
    orientation_deg : float, 默认 90
    椭圆长轴相对于 x 轴的旋转角，单位度。
    0° 表示长轴朝东西向，90° 表示长轴朝南北向。
    """
    theta = np.radians(orientation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    filt = []
    a2, b2 = a_radius ** 2, b_radius ** 2
    for x, y in positions:
        # 旋转坐标系，使椭圆对齐坐标轴
        x_rot = x * cos_t + y * sin_t
        y_rot = -x * sin_t + y * cos_t
        if (x_rot ** 2) / a2 + (y_rot ** 2) / b2 <= 1.0:
            filt.append((x, y))
    return filt

if __name__ == "__main__":
    print("=== 敦煌100MW熔盐塔式光热电站定日镜场布局可视化 ===")
    
    # 可视化最佳布局
    print("\n🎨 生成最佳布局可视化...")
    positions, best_solution = visualize_best_layout()
    
    # 比较不同方案
    print("\n📊 生成方案对比图...")
    compare_layouts()
    
    print("\n✅ 可视化完成！")