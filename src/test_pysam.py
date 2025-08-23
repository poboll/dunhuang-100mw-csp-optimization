#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试PySAM TcsmoltenSalt模块的基本功能
"""

import os
from pathlib import Path

# 设置SSC库路径
SSC_DYLIB_PATH = Path("/Applications/SAM_2025.4.16/ssc.dylib")
if SSC_DYLIB_PATH.exists():
    os.environ['SSC_DLL_PATH'] = str(SSC_DYLIB_PATH)
    print(f"✅ SSC库路径设置成功: {SSC_DYLIB_PATH}")
else:
    print(f"❌ 警告: 在指定路径中未找到 ssc.dylib")

try:
    import PySAM.TcsmoltenSalt as TcsmoltenSalt
    print("✅ PySAM.TcsmoltenSalt 导入成功")
    
    # 尝试创建一个基本实例
    case = TcsmoltenSalt.new()
    print("✅ TcsmoltenSalt 实例创建成功")
    
    # 设置最基本的必需参数
    case.SystemDesign.P_ref = 100
    case.SystemDesign.T_htf_cold_des = 290
    case.SystemDesign.T_htf_hot_des = 565
    case.SystemDesign.design_eff = 0.412
    case.SystemDesign.dni_des = 950
    case.SystemDesign.solarm = 2.4
    case.SystemDesign.tshours = 11
    print("✅ SystemDesign 参数设置成功")
    
    # 设置SystemControl必需参数
    case.SystemControl.pb_fixed_par = 0.0055
    case.SystemControl.f_turb_tou_periods = [1]*9
    
    # 简单的调度表
    schedule = [[1]*24 for _ in range(12)]
    case.SystemControl.weekday_schedule = schedule
    case.SystemControl.weekend_schedule = schedule
    print("✅ SystemControl 参数设置成功")
    
    print("\n🎉 基本PySAM测试通过!")
    
except Exception as e:
    print(f"❌ PySAM测试失败: {e}")
    import traceback
    traceback.print_exc()