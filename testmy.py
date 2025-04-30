import myapp
import os
import cv2
# myapp.main()
dataset_dir = "dataset"
result_dir = "result"

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

for filename in os.listdir(dataset_dir):
    file_path = os.path.join(dataset_dir, filename)
    if os.path.isfile(file_path) and "ph" not in filename:
        complete_html, table_boxes_img, ocr_boxes_img, _ = myapp.process_image(file_path, True, None, None, True, 5, 5, None, False)
        # complete_html是一个html文本， 另外二者是图片
        # 将三个文件都保存到result_dir目录下
        with open(os.path.join(result_dir, filename + ".html"), "w") as f:
            f.write(complete_html)
        cv2.imwrite(os.path.join(result_dir, filename + "_table_boxes.jpg"), table_boxes_img)
        cv2.imwrite(os.path.join(result_dir, filename + "_ocr_boxes.jpg"), ocr_boxes_img)
        