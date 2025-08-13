import os
from PIL import Image

# --- 1. 参数配置 ---

# 存放瓦片的文件夹名称
output_dir = "tiles"

# !!! 关键：这里的范围必须与您已下载的瓦片文件名完全对应 !!!
x_range = range(199815, 199842)
y_range = range(162950, 162975)


# --- 2. 拼接瓦片 ---

print("开始从本地文件拼接瓦片...")

if not os.path.exists(output_dir):
    print(f"错误：找不到文件夹 '{output_dir}'。")
    exit()

try:
    any_tile_path = os.path.join(output_dir, f"{x_range.start}_{y_range.start}.webp")
    with Image.open(any_tile_path) as first_tile:
        tile_width, tile_height = first_tile.size
except FileNotFoundError:
    print(f"错误：在 '{output_dir}' 文件夹中找不到瓦片文件。")
    print("请确认您的 x_range 和 y_range 设置是否正确。")
    exit()

total_width = tile_width * len(x_range)
total_height = tile_height * len(y_range)
stitched_image = Image.new('RGB', (total_width, total_height))

for i, y in enumerate(y_range):
    for j, x in enumerate(x_range):
        tile_path = os.path.join(output_dir, f"{x}_{y}.webp")
        if os.path.exists(tile_path):
            try:
                with Image.open(tile_path) as tile:
                    paste_x = j * tile_width
                    paste_y = (len(y_range) - 1 - i) * tile_height
                    stitched_image.paste(tile, (paste_x, paste_y))
            except Exception as e:
                print(f"处理瓦片 {tile_path} 时出错: {e}")

stitched_image_path = "stitched_image_full.png"
stitched_image.save(stitched_image_path)
print(f"完整拼接图像已保存至: {stitched_image_path}")


# --- 3. 裁剪最终图像 (请在这里修改) ---

print("开始裁剪图像...")

# --- 在这里修改您想从每条边裁掉的像素量 ---
# 您说要减去左边和下边一点点，可以先试试下面的值。
# 如果觉得裁少了，就把数字改大一点。
# 如果觉得裁多了，就把数字改小一点。

trim_from_left = 600    # <--- 从左边裁掉 400 像素
trim_from_bottom = 450  # <--- 从下边裁掉 400 像素

trim_from_top = 0       # 不裁剪顶部
trim_from_right = 0     # 不裁剪右边
# -----------------------------------------

# 根据您设置的裁剪量，计算最终的裁剪区域
left = trim_from_left
upper = trim_from_top
right = total_width - trim_from_right
lower = total_height - trim_from_bottom

# 检查裁剪参数是否有效
if left >= right or upper >= lower:
    print(f"错误：裁剪参数无效，裁剪过度。请减小裁剪量。")
else:
    # 执行裁剪
    crop_box = (left, upper, right, lower)
    cropped_image = stitched_image.crop(crop_box)

    # 保存最终的裁剪后大图
    cropped_image_path = "final_cropped_image.png"
    cropped_image.save(cropped_image_path)

    print(f"裁剪完成！最终图像已保存至: {cropped_image_path}")