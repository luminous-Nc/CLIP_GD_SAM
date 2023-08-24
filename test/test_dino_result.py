from PIL import Image, ImageDraw
import torch

# 假设这是图像的路径和边界框坐标
image_path = "food_pictures/juice.jpg"
cx, cy, w, h = 0.2967, 0.3106, 0.2816, 0.2098

# 打开图像
image = Image.open(image_path)

# 获取图像的宽度和高度
image_width, image_height = image.size

# 将相对坐标转换为绝对坐标
abs_cx = cx * image_width
abs_cy = cy * image_height
abs_w = w * image_width
abs_h = h * image_height

# 计算矩形框的左上角和右下角坐标
left = abs_cx - abs_w / 2
top = abs_cy - abs_h / 2
right = abs_cx + abs_w / 2
bottom = abs_cy + abs_h / 2

# 创建一个可绘制的图像副本
draw = ImageDraw.Draw(image)

# 绘制边界框
draw.rectangle([left, top, right, bottom], outline="red", width=2)

# 显示或保存绘制好的图像
image.show()
# 或者 image.save("output_image.jpg")
