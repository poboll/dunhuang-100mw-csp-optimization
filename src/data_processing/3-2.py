import cv2
import numpy as np
import csv

# 1. 加载图像
image = cv2.imread('/Users/Apple/Downloads/官方主题/final_cropped_image.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- 优化点 1: 使用自适应阈值 ---
# cv2.adaptiveThreshold(源图像, 最大值, 阈值计算方法, 阈值类型, 邻域大小, 常数C)
# ADAPTIVE_THRESH_GAUSSIAN_C: 基于高斯分布的加权平均值计算阈值，效果通常更好
# BLOCK_SIZE: 邻域大小，必须是奇数。这个值需要尝试，比如 11, 15, 21 ...
# C: 从均值或加权均值中减去的常数。可以为正、零或负。
print("正在进行自适应阈值处理...")
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 15, 4)

# 将处理后的二值图保存下来，方便调试和观察效果
cv2.imwrite('debug_adaptive_thresh.jpg', thresh)
print("自适应阈值结果已保存到 debug_adaptive_thresh.jpg")


# --- 优化点 2: 形态学开运算去噪 ---
# 定义一个结构元素（内核），这里用一个 3x3 的圆形
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
# 执行开运算
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# 将去噪后的图像也保存下来
cv2.imwrite('debug_morph_opening.jpg', opening)
print("形态学去噪结果已保存到 debug_morph_opening.jpg")


# --- 优化点 3: 精细化斑点检测器参数 ---
print("正在设置和运行斑点检测器...")
params = cv2.SimpleBlobDetector_Params()

# 核心参数：面积过滤。这个范围需要精确调整。
# 我根据您的图像估算，单个镜子的面积大约在 20 到 80 像素之间。
params.filterByArea = True
params.minArea = 20    # 调高最小面积，过滤掉绝大部分噪点
params.maxArea = 100   # 调低最大面积，避免将多个粘连的镜子误判为一个

# 核心参数：形状过滤。镜子是圆的，所以可以提高圆度要求。
params.filterByCircularity = True
params.minCircularity = 0.7  # 提高最小圆度，要求形状更接近圆

# 其他参数可以保持或略微收紧
params.filterByConvexity = True
params.minConvexity = 0.85

params.filterByInertia = True
params.minInertiaRatio = 0.4 # 惯性率，圆为1，直线为0。提高此值可确保形状不狭长。

# 创建检测器
detector = cv2.SimpleBlobDetector_create(params)

# 在去噪后的图像上执行检测
keypoints = detector.detect(opening)

# 4. 提取坐标并保存
heliostat_positions = []
for kp in keypoints:
    x, y = kp.pt
    heliostat_positions.append((int(x), int(y)))

print(f"检测到 {len(heliostat_positions)} 个定日镜。")

with open('heliostat_layout_optimized.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y'])
    writer.writerows(heliostat_positions)
print("优化后的镜场布局坐标已保存到 heliostat_layout_optimized.csv 文件。")

# 5. 可视化结果
im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('detection_result_optimized.jpg', im_with_keypoints)
print("优化后的检测结果可视化图像已保存到 detection_result_optimized.jpg。")