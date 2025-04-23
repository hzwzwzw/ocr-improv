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

# img = cv2.imread('clean_img.jpg')
# ocr_detection = pipeline(Tasks.ocr_detection, model='iic/cv_resnet18_ocr-detection-line-level_damo')
ocr_detection = pipeline(Tasks.ocr_detection, model='iic/cv_resnet18_ocr-detection-db-line-level_damo')
ocr_recognition = pipeline(Tasks.ocr_recognition, model='iic/cv_convnextTiny_ocr-recognition-general_damo')
# ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')
# table_recognition = pipeline(Tasks.table_recognition, model='iic/cv_dla34_table-structure-recognition_cycle-centernet')

def recognize(img):
    result = ocr_recognition(img)
    return result

def ocr(img):
    polygon = ocr_detection(img)
    results = []
    for box in tqdm(polygon['polygons'], desc="Processing boxes"):
        x1, y1, x2, y2 = box[0], box[1], box[4], box[5]
        if x1 == x2 or y1 == y2:
            results.append({'text': ['']})
            continue
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        cropped_img = img[y1:y2, x1:x2]
        # 拓展cropped_img，四周加上10个像素白色
        # cropped_img = cv2.copyMakeBorder(cropped_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        results.append(recognize(cropped_img))
    return polygon, results
    # print(results)

if __name__ == "__main__":
    img = cv2.imread('data/10.jpg')
    # save img
    # print(result)
    result, results = ocr(img)
    for i, box in enumerate(result['polygons']):
        x1, y1, x2, y2 = box[0], box[1], box[4], box[5]
        if x1 == x2 or y1 == y2:
            continue
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        # img = draw_box_string(img.copy(), x1, y1, results[i])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        polybox = [[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]]
        cv2.fillPoly(img, np.int32([polybox]), (255,255,255), lineType=cv2.LINE_AA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("simfang.ttf", 40, encoding="utf-8")
    for i, box in enumerate(result['polygons']):
        x1, y1, x2, y2 = box[0], box[1], box[4], box[5]
        if x1 == x2 or y1 == y2:
            continue
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        draw.text((x1, y1), results[i]['text'][0], (0, 100, 0), font=font)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # save image

    cv2.imwrite('dg.jpg', img)