import geopandas as gpd
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --- 1. Setup: Define file paths and constants ---
DNI_RASTER_PATH = 'DNI.tif'
XINJIANG_JSON_PATH = 'xinjiang.json'
CHINA_MAP_PATH = 'china_full.json'
WORLD_MAP_PATH = 'ne_110m_admin_0_countries.shp'
OUTPUT_FILENAME = 'xinjiang_dni_map_publication_final_v4.png'

# --- 2. Data Loading and Preprocessing ---
xinjiang = gpd.read_file(XINJIANG_JSON_PATH).dissolve()
with rasterio.open(DNI_RASTER_PATH) as src:
    if xinjiang.crs != src.crs:
        xinjiang = xinjiang.to_crs(src.crs)
    data, transform = mask(src, xinjiang.geometry, crop=True)
    nodata = src.nodata
    data = data.astype(float)
    if nodata is not None:
        data[data == nodata] = np.nan

# compute plot extent in native CRS units
xmin = transform[2]
xmax = xmin + transform[0] * data.shape[2]
ymin = transform[5] + transform[4] * data.shape[1]
ymax = transform[5]
extent = [xmin, xmax, ymin, ymax]

# --- 3. Visualization: Main Map ---
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})
fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=False)
# Shift main map left further to make room for lower-right inset
grid_pos = ax.get_position()
ax.set_position([grid_pos.x0, grid_pos.y0, grid_pos.width * 0.75, grid_pos.height])  # Shrink main map more to the right to accommodate larger inset

# Use perceptually uniform sequential colormap (Cividis is blue-to-yellow, print- & colorblind-friendly)
cmap = plt.cm.get_cmap('cividis')  # If preferred, alternatives: 'viridis', 'magma', or 'inferno'
img = ax.imshow(data.squeeze(),
                cmap=cmap,
                extent=extent,
                vmin=np.nanpercentile(data, 2),
                vmax=np.nanpercentile(data, 98))
# Xinjiang boundary outline
xinjiang.boundary.plot(ax=ax, edgecolor='black', linewidth=1.2)

# Remove axes
ax.set_axis_off()

# Colorbar
cbar = fig.colorbar(img, ax=ax, shrink=0.7, aspect=25, pad=0.02)
cbar.set_label('Average Daily DNI (kWh/m²/day)', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# Scale bar & North arrow unchanged
scale_km = 400
km_per_deg = 84
scale_deg = scale_km / km_per_deg
sb_x = extent[1] - scale_deg * 1.6
sb_y = extent[2] + (extent[3] - extent[2]) * 0.05
ax.plot([sb_x, sb_x + scale_deg], [sb_y, sb_y], 'k-', lw=2)
ax.text(sb_x + scale_deg / 2, sb_y + 0.015*(ymax-ymin), f'{scale_km} km',
        ha='center', va='bottom', fontsize=11)
ax.annotate('N', xy=(0.95, 0.90), xytext=(0.95, 0.82),
            arrowprops=dict(facecolor='k', width=4, headlength=8),
            ha='center', va='center', fontsize=14, fontweight='bold',
            xycoords=ax.transAxes)

# --- 4. Inset Map: China with full territory ---
# Use a figure-level inset at fixed figure coordinates for true left flush
# Create a more compact inset at figure-left
axins = fig.add_axes([0.01, 0.60, 0.30, 0.30], facecolor='#eaf8ff')  # [left, bottom, width, height]
axins.patch.set_edgecolor('black')
axins.patch.set_linewidth(0.7)  # slimmer border  # [left, bottom, width, height] in figure fraction

# Plot Asia background
world = gpd.read_file(WORLD_MAP_PATH)
asia = world[world['CONTINENT'] == 'Asia']
asia.plot(ax=axins, color='white', edgecolor='gray', linewidth=0.5)

# Plot full China and highlight Xinjiang
china = gpd.read_file(CHINA_MAP_PATH)
china.plot(ax=axins, color='lightgray', edgecolor='black', linewidth=0.7)
xinjiang.to_crs(china.crs).plot(ax=axins, color='red')

# Inset styling
axins.patch.set_edgecolor('black')
axins.patch.set_linewidth(1.0)
for spine in axins.spines.values():
    spine.set_linewidth(1.0)
# Include full Xinjiang and surrounding context
axins.set_xlim(70, 140)
axins.set_ylim(0, 54)
axins.set_xticks([])
axins.set_yticks([])

# --- 5. Save and show ---
fig.savefig(OUTPUT_FILENAME, dpi=600)
plt.show()
