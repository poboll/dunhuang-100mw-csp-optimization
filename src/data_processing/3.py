import requests
import os
from PIL import Image
from io import BytesIO
import time

# --- 1. 参数配置 ---

# 从您的URL中获取的token
token = "Bearer%20fdsa0c81f784490a4c5dfghdfgh" 

# 瓦片URL模板
# 注意：这里的 {x} 和 {y} 是占位符，将在循环中被替换
url_template = "https://tile.charmingglobe.com/tile/china2023_5_shield/tms/18/{x}/{y}?v=v1&token={token}"

# 定义您要下载的瓦片坐标范围
# 您需要根据实际需求调整这些值
x_range = range(199815, 199842)  # 示例 X 坐标范围
y_range = range(162950, 162975)  # 示例 Y 坐标范围

# 定义请求头，模仿浏览器行为
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# --- 2. 下载瓦片 ---

# 创建一个目录来保存下载的瓦片
output_dir = "tiles"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("开始下载瓦片...")

for y in y_range:
    for x in x_range:
        # 构建完整的瓦片URL
        tile_url = url_template.format(x=x, y=y, token=token)
        
        # 定义本地保存的文件路径
        file_path = os.path.join(output_dir, f"{x}_{y}.webp")
        
        # 如果文件已存在，则跳过下载
        if os.path.exists(file_path):
            print(f"文件 {file_path} 已存在，跳过。")
            continue

        try:
            # 发送HTTP请求
            response = requests.get(tile_url, headers=headers)
            # 检查请求是否成功
            if response.status_code == 200:
                # 将图片内容写入文件
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"成功下载: {file_path}")
            else:
                print(f"下载失败: {tile_url} (状态码: {response.status_code})")

            # !!! 重要：在每个请求后添加一个小的延迟，以避免给服务器造成过大压力
            time.sleep(0.5)  # 延迟0.5秒

        except requests.exceptions.RequestException as e:
            print(f"下载时发生错误: {e}")

print("所有瓦片下载完成！")


# --- 3. 拼接瓦片 ---

print("开始拼接瓦片...")

# 假设所有瓦片的大小都相同，我们先获取一个瓦片的尺寸
# 注意：为了代码能运行，您需要保证至少有一个瓦片被成功下载
try:
    first_tile_path = os.path.join(output_dir, f"{x_range.start}_{y_range.start}.webp")
    with Image.open(first_tile_path) as first_tile:
        tile_width, tile_height = first_tile.size
except FileNotFoundError:
    print("错误：找不到起始瓦片，无法确定瓦片尺寸。请确保至少下载了一个瓦片。")
    exit()

# 计算最终大图的尺寸
total_width = tile_width * len(x_range)
total_height = tile_height * len(y_range)

# 创建一个空白的大图
stitched_image = Image.new('RGB', (total_width, total_height))

# 循环遍历每个瓦片并将其粘贴到大图的正确位置
for i, y in enumerate(y_range):
    for j, x in enumerate(x_range):
        tile_path = os.path.join(output_dir, f"{x}_{y}.webp")
        if os.path.exists(tile_path):
            try:
                tile = Image.open(tile_path)
                # 计算粘贴位置
                # 请注意：地图瓦片的Y坐标通常是从上到下增加的，
                # 所以我们使用 (len(y_range) - 1 - i) 来反转Y轴的粘贴顺序。
                paste_x = j * tile_width
                paste_y = (len(y_range) - 1 - i) * tile_height
                stitched_image.paste(tile, (paste_x, paste_y))
            except Exception as e:
                print(f"拼接瓦片 {tile_path} 时出错: {e}")

# 保存最终的拼接大图
stitched_image_path = "stitched_satellite_image.png"
stitched_image.save(stitched_image_path)

print(f"拼接完成！最终图像已保存至: {stitched_image_path}")

