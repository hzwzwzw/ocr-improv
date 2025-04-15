
import cv2
from rapidocr_onnxruntime import RapidOCR
from PIL import Image, ImageDraw, ImageFont
import numpy as np

picpath = "images/10.jpg"
img = cv2.imread(picpath)
def draw_box_string(img, x, y, string):
    """
    img: imread读取的图片;
    x,y:字符起始绘制的位置;
    string: 显示的文字;
    return: img
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    # simhei.ttf 是字体，你如果没有字体，需要下载
    font = ImageFont.truetype("simfang.ttf", 50, encoding="utf-8")
    draw.text((x, y), string, (0, 100, 0), font=font)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img

# 预处理： 关键改进部分
# 图像是扫描件，对其进行滤波、去噪、二值化等预处理
# 转为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 高斯滤波去噪
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# 二值化
_, img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 保存预处理后的图像
cv2.imwrite("images/10_preprocessed.jpg", img)
# 识别
model = RapidOCR()
result = model(img, det=True, rec=True, cls=True)
# 结果
result = result[0]
value = [x[1] for x in result]
print(value)
# 在图像上绘制识别到的方框
for x in result:
    box = x[0]
    # print(box)
    # box belike [[2147, 184], [2310, 184], [2310, 238], [2147, 238]]
    cv2.rectangle(img, (int(box[0][0]), int(box[0][1])), (int(box[2][0]), int(box[2][1])), (0, 255, 0), 2)
    # 方框内设为白色
    cv2.fillPoly(img, [np.array(box).astype(int)], (255, 255, 255))
    # 绘制文本，包含中文
    img = draw_box_string(img, int(box[0][0]), int(box[0][1]), x[1])

# 显示识别结果
cv2.imwrite("images/10_result.jpg", img)
