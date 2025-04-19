# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from PIL import Image
# import requests

# # load image from the IAM database (actually this model is meant to be used on printed text)
# # url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
# # image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# image = Image.open("snip.png")

# processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
# model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
# pixel_values = processor(images=image, return_tensors="pt").pixel_values

# generated_ids = model.generate(pixel_values)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# print(generated_text)

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2
from tqdm import tqdm
# import ocrtest
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# model_rec = "modeltest/cv_convnextTiny_ocr-recognition-document_damo/model.onnx"
# model_det = "modeltest/cv_resnet18_ocr-detection-line-level_damo/model.onnx"

img = cv2.imread('data/10.jpg')
ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')

ocr_detection = pipeline(Tasks.ocr_detection, model='iic/cv_resnet18_ocr-detection-line-level_damo')
result = ocr_detection(img)
# save img
# print(result)
results = []

def recognize(img):
    result = ocr_recognition(img)
    return result
for box in tqdm(result['polygons'], desc="Processing boxes"):
    x1, y1, x2, y2 = box[0], box[1], box[4], box[5]
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    cropped_img = img[y1:y2, x1:x2]
    results.append(recognize(cropped_img))

print(results)

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

# draw results
for i, box in enumerate(result['polygons']):
    x1, y1, x2, y2 = box[0], box[1], box[4], box[5]
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    img = draw_box_string(img, x1, y1, results[i])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite('dg.jpg', img)