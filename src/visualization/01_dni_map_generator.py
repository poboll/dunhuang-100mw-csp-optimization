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
from shapely.geometry import Point, box

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
gansu.boundary.plot(ax=ax, edgecolor='black', linewidth=1.2)

# Remove grid lines
ax.grid(False)
ax.set_axisbelow(True)

# 突出显示敦煌市（只显示边框）
dunhuang.boundary.plot(ax=ax, edgecolor='red', linewidth=2.5)

# Add Dunhuang city annotation
dunhuang_center = dunhuang.geometry.centroid.iloc[0]
ax.annotate('Dunhuang City', 
            xy=(dunhuang_center.x, dunhuang_center.y),
            xytext=(80, -60), textcoords='offset points',
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

# Add scale bar (convert degrees to approximate km) - positioned at coordinates (107, 32.5)
scale_length_deg = 2.0  # 2 degrees
scale_length_km = scale_length_deg * 111  # Approximate km per degree (222 km is correct)
scale_x = 107  # Longitude position
scale_y = 32.5   # Latitude position (moved up slightly)
ax.plot([scale_x, scale_x + scale_length_deg], [scale_y, scale_y], 'k-', linewidth=2)
ax.text(scale_x + scale_length_deg/2, scale_y - 0.3, 
        f'{scale_length_km:.0f} km', ha='center', fontsize=9)

# Add enhanced north arrow with thicker lines
north_x = extent[1] - (extent[1] - extent[0]) * 0.08
north_y = extent[3] - (extent[3] - extent[2]) * 0.12  # Moved down
arrow_length = (extent[3] - extent[2]) * 0.035  # Slightly longer arrow

# Draw arrow shaft with thicker line
ax.plot([north_x, north_x], [north_y - arrow_length, north_y], 'k-', linewidth=3)

# Draw arrow head with thicker lines
head_width = (extent[1] - extent[0]) * 0.012
head_length = (extent[3] - extent[2]) * 0.012
ax.arrow(north_x, north_y - head_length, 0, head_length, 
         head_width=head_width, head_length=head_length, 
         fc='black', ec='black', linewidth=2)

# Add 'N' label with enhanced styling
ax.annotate('N', xy=(north_x, north_y + (extent[3] - extent[2]) * 0.015), 
            fontsize=14, fontweight='bold', ha='center', va='bottom', color='black')

# --- 4. Inset Map: China with Gansu Province Highlighted ---
# Create inset map above scale bar, slightly longer and with ocean background
axins = fig.add_axes([0.10, 0.10, 0.42, 0.37])
axins.patch.set_facecolor('#E6F3FF')  # Light blue for ocean background
axins.patch.set_edgecolor('black')
axins.patch.set_linewidth(1.0)

# Load world map data
world_map_path = '/Users/Apple/Downloads/官方主题/data/spatial/ne_110m_admin_0_countries.shp'
world = gpd.read_file(world_map_path)

# Ensure CRS consistency
if world.crs != china.crs:
    world = world.to_crs(china.crs)

# Get China bounds for setting map extent
china_bounds = china.total_bounds
padding = (china_bounds[2] - china_bounds[0]) * 0.05  # Minimal padding to maximize China display

# Define the extent for the inset map (slightly larger than China)
map_extent = [
    china_bounds[0] - padding,
    china_bounds[2] + padding,
    china_bounds[1] - padding,
    china_bounds[3] + padding
]

# Clip world data to the map extent
clip_box = box(map_extent[0], map_extent[2], map_extent[1], map_extent[3])
world_clipped = world.clip(clip_box)

# Separate China from other countries
china_from_world = world_clipped[world_clipped['NAME'].isin(['China', 'People\'s Republic of China'])]
neighboring_countries = world_clipped[~world_clipped['NAME'].isin(['China', 'People\'s Republic of China'])]

# Plot in correct order: ocean background (already set), neighboring countries, China, provinces, Gansu

# 1. Plot neighboring countries with white fill and gray borders
if not neighboring_countries.empty:
    neighboring_countries.plot(ax=axins, color='white', edgecolor='gray', linewidth=0.5, alpha=1.0)

# 2. Plot China with light gray fill and black borders
china.plot(ax=axins, color='lightgray', edgecolor='black', linewidth=0.5)

# 3. Plot China's provincial boundaries in black (thinner lines)
china.boundary.plot(ax=axins, edgecolor='black', linewidth=0.4)

# 4. Highlight Gansu Province
gansu.plot(ax=axins, color='red', alpha=0.7)

# Set inset map extent to fill the entire axes area (no padding to eliminate gap)
axins.set_xlim(map_extent[0], map_extent[1])
axins.set_ylim(map_extent[2], map_extent[3])
axins.set_xticks([])
axins.set_yticks([])

# --- 5. Save high-quality image for scientific publication ---
# Save with high DPI (600) for scientific journal quality
fig.savefig(OUTPUT_FILENAME, dpi=600, bbox_inches='tight', format='png')
print(f"图片已保存至: {OUTPUT_FILENAME}")
# plt.show()  # 注释掉显示，避免阻塞终端
