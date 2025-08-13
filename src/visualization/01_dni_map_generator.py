#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solar resource distribution and heliostat field optimization in Dunhuang, China (2024)

Data Sources:
1. Solar Resource Data: Global Solar Atlas (https://globalsolaratlas.info/)
   - DNI data from Solargis satellite-based model (v2.1)
   - Based on Meteosat, GOES, MTSAT, Himawari-8 satellite imagery
   - Atmospheric data from ECMWF MACC-II/CAMS
   
2. Meteorological Data: NREL Typical Meteorological Year (TMY)
   - Location: 40.063°N, 94.426°E, Elevation: 1267m
   
3. Geographic Boundaries: Administrative boundary vector data
   - China, Gansu Province, Dunhuang City boundaries
   
4. Satellite Imagery: CharmingGlobe 2023 China high-resolution tiles

Software: Python (v3.10) with Matplotlib (v3.4.0), GeoPandas (v0.9.0), Rasterio (v1.2.0)
Available at: https://www.python.org, https://matplotlib.org, https://geopandas.org, https://rasterio.readthedocs.io
"""

import geopandas as gpd
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point

# --- 1. Setup: Define file paths and constants ---
DNI_RASTER_PATH = 'gee/DNI.tif'
GANSU_JSON_PATH = 'data/spatial/gansu.json'
DUNHUANG_JSON_PATH = 'data/spatial/dunhuang.json'
CHINA_MAP_PATH = 'data/spatial/china_full.json'
OUTPUT_FILENAME = 'results/gansu_dni_map_with_dunhuang_v1.png'

# --- 2. Data Loading and Preprocessing ---
gansu = gpd.read_file(GANSU_JSON_PATH)
dunhuang = gpd.read_file(DUNHUANG_JSON_PATH)
china = gpd.read_file(CHINA_MAP_PATH)

# 读取DNI栅格数据并裁剪到甘肃省范围
with rasterio.open(DNI_RASTER_PATH) as src:
    # 确保坐标系一致
    if gansu.crs != src.crs:
        gansu = gansu.to_crs(src.crs)
    if dunhuang.crs != src.crs:
        dunhuang = dunhuang.to_crs(src.crs)
    if china.crs != src.crs:
        china = china.to_crs(src.crs)
    
    # 裁剪DNI数据到甘肃省范围
    data, transform = mask(src, gansu.geometry, crop=True)
    nodata = src.nodata
    data = data.astype(float)
    if nodata is not None:
        data[data == nodata] = np.nan
    
    # 计算绘图范围
    xmin = transform[2]
    xmax = xmin + transform[0] * data.shape[2]
    ymin = transform[5] + transform[4] * data.shape[1]
    ymax = transform[5]
    extent = [xmin, xmax, ymin, ymax]

# --- 3. Visualization: Main Map ---
# Set scientific journal standard fonts
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.unicode_minus': False
})

fig, ax = plt.subplots(figsize=(10.8, 7.2), constrained_layout=True)

# 显示DNI数据（取第一个波段）
if len(data.shape) == 3:
    data_2d = data[0]  # 取第一个波段
else:
    data_2d = data

# Use colormap suitable for scientific publications
cmap = plt.colormaps['YlOrRd']  # Yellow-Orange-Red colormap, suitable for DNI data
img = ax.imshow(data_2d,
                cmap=cmap,
                extent=extent,
                origin='upper',
                vmin=np.nanpercentile(data_2d, 5),
                vmax=np.nanpercentile(data_2d, 95),
                alpha=0.8)

# Add padding around the map by adjusting axis limits
padding_x = (extent[1] - extent[0]) * 0.1
padding_y = (extent[3] - extent[2]) * 0.1
ax.set_xlim(extent[0] - padding_x, extent[1] + padding_x)
ax.set_ylim(extent[2] - padding_y, extent[3] + padding_y)

# 绘制甘肃省边界
gansu.boundary.plot(ax=ax, edgecolor='black', linewidth=2)

# Remove grid lines
ax.grid(False)
ax.set_axisbelow(True)

# 突出显示敦煌市（只显示边框）
dunhuang.boundary.plot(ax=ax, edgecolor='red', linewidth=2.5)

# Add Dunhuang city annotation
dunhuang_center = dunhuang.geometry.centroid.iloc[0]
ax.annotate('Dunhuang City', 
            xy=(dunhuang_center.x, dunhuang_center.y),
            xytext=(60, -40), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black', linewidth=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='black', lw=1.5),
            fontsize=18, fontweight='normal', ha='center')

# Set axis labels (no title for scientific journals)
ax.set_xlabel('Longitude (°E)', fontsize=10)
ax.set_ylabel('Latitude (°N)', fontsize=10)

# Add colorbar
cbar = fig.colorbar(img, ax=ax, shrink=0.8, aspect=25, pad=0.02)
cbar.set_label('DNI (kWh/m²/day)', fontsize=10)
cbar.ax.tick_params(labelsize=9)

# Add scale bar (convert degrees to approximate km) - positioned at coordinates (107, 33)
scale_length_deg = 2.0  # 2 degrees
scale_length_km = scale_length_deg * 111  # Approximate km per degree (222 km is correct)
scale_x = 107  # Longitude position
scale_y = 33   # Latitude position (adjusted down from 34)
ax.plot([scale_x, scale_x + scale_length_deg], [scale_y, scale_y], 'k-', linewidth=2)
ax.text(scale_x + scale_length_deg/2, scale_y - 0.3, 
        f'{scale_length_km:.0f} km', ha='center', fontsize=9)

# Add north arrow
north_x = extent[1] - (extent[1] - extent[0]) * 0.08
north_y = extent[3] - (extent[3] - extent[2]) * 0.08
arrow_length = (extent[3] - extent[2]) * 0.025
ax.arrow(north_x, north_y - arrow_length, 0, arrow_length, 
         head_width=(extent[1] - extent[0]) * 0.008, head_length=(extent[3] - extent[2]) * 0.008, 
         fc='black', ec='black', linewidth=1)
ax.annotate('N', xy=(north_x, north_y + (extent[3] - extent[2]) * 0.01), 
            fontsize=12, fontweight='bold', ha='center', va='bottom')

# --- 4. Inset Map: China with Gansu Province Highlighted ---
# Create inset map above scale bar, slightly larger size
axins = fig.add_axes([0.10, 0.15, 0.37, 0.37], facecolor='white')
axins.patch.set_edgecolor('black')
axins.patch.set_linewidth(1.0)

# Plot China map
china.plot(ax=axins, color='lightgray', edgecolor='black', linewidth=0.5)

# Highlight Gansu Province
gansu.plot(ax=axins, color='red', alpha=0.7, edgecolor='darkred', linewidth=1.0)

# Set inset map extent and style (no title, no labels for scientific journals)
china_bounds = china.total_bounds
padding = (china_bounds[2] - china_bounds[0]) * 0.02
axins.set_xlim(china_bounds[0] - padding, china_bounds[2] + padding)
axins.set_ylim(china_bounds[1] - padding, china_bounds[3] + padding)
axins.set_xticks([])
axins.set_yticks([])

# --- 5. Save and show ---
fig.savefig(OUTPUT_FILENAME, dpi=600)
plt.show()
