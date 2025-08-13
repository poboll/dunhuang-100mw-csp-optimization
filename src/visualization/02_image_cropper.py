from PIL import Image
import numpy as np

# 设置输入输出路径
input_path = "results/gansu_dni_map_with_dunhuang_v1.png"
output_path = "results/gansu_dni_map_with_dunhuang_v1_cropped.png"

# 读取图像
img = Image.open(input_path)
img_np = np.array(img)

# 转为灰度判断非白区域
gray = np.mean(img_np[:, :, :3], axis=2)
non_white_threshold = 250  # 将接近白色的区域视为空白
mask = gray < non_white_threshold

# 获取非空白区域的边界
rows = np.any(mask, axis=1)
cols = np.any(mask, axis=0)
top = np.argmax(rows)
bottom = len(rows) - np.argmax(rows[::-1])
left = np.argmax(cols)
right = len(cols) - np.argmax(cols[::-1])

# 设置边距（单位：像素）
padding_top = 40
padding_bottom = 30  # 特别指定底部边距为 30
padding_left = 80   # 增加左边距以裁剪更多左侧白边
padding_right = 40

# 应用边距并确保不越界
top_new = max(top - padding_top, 0)
bottom_new = min(bottom + padding_bottom, img_np.shape[0])
left_new = max(left - padding_left, 0)
right_new = min(right + padding_right, img_np.shape[1])

# 裁剪图像
cropped_img_np = img_np[top_new:bottom_new, left_new:right_new]
cropped_img = Image.fromarray(cropped_img_np)

# 保存图像，设置 600 DPI
cropped_img.save(output_path, dpi=(600, 600))
print(f"裁剪后的图像已保存至: {output_path}")
