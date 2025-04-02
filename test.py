import app
import os

# test cases are in data directory
files = os.listdir("data")
for file in files:
    result = app.process_image(file, )
    html_output, table_boxes_output, ocr_boxes_output, elapse_text = result