#!/usr/bin/env python3
# PySAM SolarPILOT 简化测试 - macOS版本

import os
from pathlib import Path
import json

# 设置SAM路径
SSC_DYLIB_PATH = Path("/Applications/SAM_2025.4.16/ssc.dylib")
if SSC_DYLIB_PATH.exists():
    os.environ['SSC_DLL_PATH'] = str(SSC_DYLIB_PATH)
    print(f"✅ 成功指定SSC核心路径: {SSC_DYLIB_PATH}")
else:
    print(f"❌ 警告: 在指定路径中未找到 ssc.dylib")

# 导入PySAM
import PySAM.Solarpilot as Solarpilot

def create_dunhuang_case():
    """
    创建一个简化的敦煌100MW电站PySAM实例。
    只设置最基本的参数以验证PySAM功能。
    """
    sp = Solarpilot.new()
    
    print("=== 设置基本系统参数 ===")
    
    # 只设置最基本的参数
    sp.SolarPILOT.rec_height = 21.6  # 接收器高度 (m)
    sp.SolarPILOT.h_tower = 260  # 塔架高度 (m)
    sp.SolarPILOT.q_design = 670  # 设计热功率 (MWt)
    
    print("✅ 基本参数设置完成")
    print(f"   - 接收器高度: {sp.SolarPILOT.rec_height} m")
    print(f"   - 塔架高度: {sp.SolarPILOT.h_tower} m")
    print(f"   - 设计热功率: {sp.SolarPILOT.q_design} MWt")
    
    print("✅ 敦煌100MW基础案例已成功创建。")
    return sp

def main():
    print("--- PySAM SolarPILOT API 测试开始 (macOS) ---")
    
    # 创建基础案例
    dunhuang_case = create_dunhuang_case()
    
    print("\n--- 参数动态修改演示 ---")
    original_height = dunhuang_case.SolarPILOT.rec_height
    print(f"原始接收器高度: {original_height} m")
    dunhuang_case.SolarPILOT.rec_height = 25.0
    print(f"修改后接收器高度: {dunhuang_case.SolarPILOT.rec_height} m")
    
    print("\n--- 基本功能验证完成 ---")
    print("✅ PySAM SolarPILOT 基本功能正常工作！")
    print("\n注意: 完整的仿真计算需要更多参数配置。")
    print("本测试主要验证PySAM的基本加载和参数设置功能。")
    
    # 导出当前配置供参考
    try:
        with open("basic_pysam_config.json", "w") as f:
            json.dump(dunhuang_case.export(), f, indent=2)
        print("\n✅ 基本配置已保存至 'basic_pysam_config.json'")
    except Exception as e:
        print(f"\n⚠️ 保存配置时出错: {e}")
    
    print("\n--- 测试结束 ---")

if __name__ == "__main__":
    main()