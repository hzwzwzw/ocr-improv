
import cv2
from rapidocr_onnxruntime import RapidOCR
from PIL import Image, ImageDraw, ImageFont
import numpy as np

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
    # simfang.ttf
    font = ImageFont.truetype("simfang.ttf", 50, encoding="utf-8")
    draw.text((x, y), string, (0, 100, 0), font=font)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


def proc(img):
    # 预处理： 关键改进部分
    # 图像是扫描件，对其进行滤波、去噪、二值化等预处理
    # 转为灰度图
    # 1. 灰度化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2. 高斯模糊
    img = cv2.GaussianBlur(img, (7, 7), 0)
    # 3. 二值化
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)
    # 4. 腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.erode(img, kernel, iterations=1)
    # 5. 膨胀
    img = cv2.dilate(img, kernel, iterations=1)
    # 6. 反色
    # img = cv2.bitwise_not(img)
    # 7. 归一化
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    # 8. 直方图均衡化
    img = cv2.equalizeHist(img)
    # 9. 颜色空间转换
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # 10. 反色
    # img = cv2.bitwise_not(img)
    # 11. 对比度，加粗
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    return img

def testocr(img):
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
    return img

if __name__ == '__main__':
    picpath = "images/10.jpg"
    img = cv2.imread(picpath)
    img = proc(img)
    img = testocr(img)
    # 保存
    cv2.imwrite("images/10_result.jpg", img)
