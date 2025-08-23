# file: src/pysam_interface.py

import os
from pathlib import Path
import json

# 导入PySAM之前先设置好路径
SSC_DYLIB_PATH = Path("/Applications/SAM_2025.4.16/ssc.dylib")
if SSC_DYLIB_PATH.exists():
    os.environ['SSC_DLL_PATH'] = str(SSC_DYLIB_PATH)
else:
    print(f"❌ 警告: 在指定路径中未找到 ssc.dylib")

import PySAM.Solarpilot as SolarPILOT

def create_full_dunhuang_case(weather_file_path: Path):
    """
    创建一个配置了所有必要参数的、可执行年度仿真的PySAM实例。
    """
    # 使用模型：SolarPILOT（仅计算光场性能，避免段错误）
    case = SolarPILOT.new()

    # --- 气候与太阳位置 ---
    case.SolarResource.solar_resource_file = str(weather_file_path)

    # --- 场地和布局 ---
    # SolarPILOT的布局参数现在在HeliostatField子组中
    case.HeliostatField.land_max = 9.5
    case.HeliostatField.land_min = 0.75
    
    # --- 系统设计 ---
    case.SystemDesign.opt_model = 0
    case.SystemDesign.p_ref = 100 # 额定功率 (MWe)
    case.SystemDesign.gross_net_conversion_factor = 0.9 # Example value, check docs

    # --- 塔和吸热器 ---
    case.TowerAndReceiver.h_tower = 260
    case.TowerAndReceiver.rec_height = 21.6 # This is the receiver height, not optical height
    case.TowerAndReceiver.D_rec = 17.65
    case.TowerAndReceiver.rec_absorptance = 0.94
    case.TowerAndReceiver.rec_type = 2 # External Cylinder
    case.HeliostatField.peak_flux_limit = 1000

    # --- 定日镜场 ---
    case.HeliostatField.helio_width = 12.2
    case.HeliostatField.helio_height = 12.2
    case.HeliostatField.helio_optical_error = 0.004
    case.HeliostatField.helio_reflectance = 0.94 * 0.95

    # --- 成本 ---
    # 映射到新的成本结构
    case.SystemCosts.tower_fixed_cost = 30e6 # 示例值, 30M$
    case.SystemCosts.rec_cost_per_area = 0 # Placeholder if needed
    case.SystemCosts.heliostat_cost = 145 # $/m2
    
    # --- 其他在文档中找到的关键参数 ---
    case.TouTranslator.weekday_schedule = [[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]] * 12
    case.TouTranslator.weekend_schedule = [[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]] * 12

    return case

def evaluate_fitness(layout_params: dict):
    """
    核心评估函数：接收布局变量，返回三个目标函数值。
    """
    weather_file = Path(__file__).parent.parent.parent / "simple_solar_resource.csv"
    
    case = create_full_dunhuang_case(weather_file)
    
    # 应用传入的布局参数
    for key, value in layout_params.items():
        # 参数现在在HeliostatField子组
        setattr(case.HeliostatField, key, value)
        
    # 执行仿真
    try:
        case.execute()
        # 输出映射
        results = {
            'f1_eff': case.Outputs.annual_total_energy_delivered / case.Outputs.annual_field_incident_energy, # 计算效率
            'f2_cost': case.Outputs.total_installed_cost,
            'f3_flux': case.Outputs.flux_max_observed
        }
        return results
    except Exception as e:
        # PySAM执行可能会抛出带内部消息的AttributeError
        print(f"❌ 仿真执行失败: {e}")
        if hasattr(case, 'Outputs') and hasattr(case.Outputs, 'errors') and case.Outputs.errors:
             print("SAM Simulation Errors:", case.Outputs.errors)
        return None