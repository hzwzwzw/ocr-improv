import time

import cv2
import gradio as gr
from lineless_table_rec import LinelessTableRecognition
from paddleocr import PPStructure
from rapid_table import RapidTable
from rapidocr_onnxruntime import RapidOCR
from table_cls import TableCls
from wired_table_rec import WiredTableRecognition

from utils import plot_rec_box, LoadImage, format_html, box_4_2_poly_to_box_4_1
img_loader = LoadImage()
table_rec_path = "models/table_rec/ch_ppstructure_mobile_v2_SLANet.onnx"
det_model_dir = {
    "mobile_det": "models/ocr/ch_PP-OCRv4_det_infer.onnx",
}

rec_model_dir = {
    "mobile_rec": "models/ocr/ch_PP-OCRv4_rec_infer.onnx",
}
table_engine_list = [
    "auto",
    "RapidTable(SLANet)",
    "RapidTable(SLANet-plus)",
    "wired_table_v2",
    "pp_table",
    "wired_table_v1",
    "lineless_table"
]

# 示例图片路径
example_images = [
    "images/wired1.png",
    "images/wired2.png",
    "images/wired3.png",
    "images/lineless1.png",
    "images/wired4.jpg",
    "images/lineless2.png",
    "images/wired5.jpg",
    "images/lineless4.jpg",
    "images/wired7.jpg",
    "images/wired9.jpg",
]
rapid_table_engine = RapidTable(model_path=table_rec_path)
SLANet_plus_table_Engine = RapidTable()
wired_table_engine_v1 = WiredTableRecognition(version="v1")
wired_table_engine_v2 = WiredTableRecognition(version="v2")
lineless_table_engine = LinelessTableRecognition()
table_cls = TableCls(model_type="q")
ocr_engine_dict = {}
pp_engine_dict = {}
for det_model in det_model_dir.keys():
    for rec_model in rec_model_dir.keys():
        det_model_path = det_model_dir[det_model]
        rec_model_path = rec_model_dir[rec_model]
        key = f"{det_model}_{rec_model}"
        ocr_engine_dict[key] = RapidOCR(det_model_path=det_model_path, rec_model_path=rec_model_path)
        pp_engine_dict[key] = PPStructure(
            layout=False,
            show_log=False,
            table=True,
            use_onnx=True,
            table_model_dir=table_rec_path,
            det_model_dir=det_model_path,
            rec_model_dir=rec_model_path
        )


def select_ocr_model(det_model, rec_model):
    return ocr_engine_dict[f"{det_model}_{rec_model}"]


def select_table_model(img, table_engine_type, det_model, rec_model):
    if table_engine_type == "RapidTable(SLANet)":
        return rapid_table_engine, table_engine_type
    elif table_engine_type == "RapidTable(SLANet-plus)":
        return SLANet_plus_table_Engine, table_engine_type
    elif table_engine_type == "wired_table_v1":
        return wired_table_engine_v1, table_engine_type
    elif table_engine_type == "wired_table_v2":
        print("使用v2 wired table")
        return wired_table_engine_v2, table_engine_type
    elif table_engine_type == "lineless_table":
        return lineless_table_engine, table_engine_type
    elif table_engine_type == "pp_table":
        return pp_engine_dict[f"{det_model}_{rec_model}"], 0
    elif table_engine_type == "auto":
        cls, elasp = table_cls(img)
        if cls == 'wired':
            table_engine = wired_table_engine_v2
            return table_engine, "wired_table_v2"
        return lineless_table_engine, "lineless_table"


def process_image(img, table_engine_type, det_model, rec_model):
    img = img_loader(img)
    start = time.time()
    table_engine, talbe_type = select_table_model(img, table_engine_type, det_model, rec_model)
    ocr_engine = select_ocr_model(det_model, rec_model)

    if isinstance(table_engine, PPStructure):
        result = table_engine(img, return_ocr_result_in_table=True)
        html = result[0]['res']['html']
        polygons = result[0]['res']['cell_bbox']
        polygons = [[polygon[0], polygon[1], polygon[4], polygon[5]] for polygon in polygons]
        ocr_boxes = result[0]['res']['boxes']
        all_elapse = f"- `table all cost: {time.time() - start:.5f}`"
    else:
        ocr_res, ocr_infer_elapse = ocr_engine(img)
        det_cost, cls_cost, rec_cost = ocr_infer_elapse
        ocr_boxes = [box_4_2_poly_to_box_4_1(ori_ocr[0]) for ori_ocr in ocr_res]
        if isinstance(table_engine, RapidTable):
            html, polygons, table_rec_elapse = table_engine(img, ocr_result=ocr_res)
            polygons = [[polygon[0], polygon[1], polygon[4], polygon[5]] for polygon in polygons]
        elif isinstance(table_engine, (WiredTableRecognition, LinelessTableRecognition)):
            html, table_rec_elapse, polygons, _, _ = table_engine(img, ocr_result=ocr_res)

        sum_elapse = time.time() - start
        all_elapse = f"- table_type: {talbe_type}\n table all cost: {sum_elapse:.5f}\n - table rec cost: {table_rec_elapse:.5f}\n - ocr cost: {det_cost + cls_cost + rec_cost:.5f}"

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    table_boxes_img = plot_rec_box(img.copy(), polygons)
    ocr_boxes_img = plot_rec_box(img.copy(), ocr_boxes)
    complete_html = format_html(html)

    return complete_html, table_boxes_img, ocr_boxes_img, all_elapse


def main():
    det_models_labels = list(det_model_dir.keys())
    rec_models_labels = list(rec_model_dir.keys())

    with gr.Blocks(css="""
        .scrollable-container {
            overflow-x: auto;
            white-space: nowrap;
        }
    """) as demo:
        with gr.Row():  # 两列布局
            with gr.Tab("Options"):
                with gr.Column(variant="panel", scale=1):  # 侧边栏，宽度比例为1
                    img_input = gr.Image(label="Upload or Select Image",  sources="upload", value="images/lineless3.jpg")

                    # 示例图片选择器
                    examples = gr.Examples(
                        examples=example_images,
                        examples_per_page=len(example_images),
                        inputs=img_input,
                        fn=lambda x: x,  # 简单返回图片路径
                        outputs=img_input,
                        cache_examples=False
                    )

                    table_engine_type = gr.Dropdown(table_engine_list, label="Select Recognition Table Engine",
                                                    value=table_engine_list[0])
                    det_model = gr.Dropdown(det_models_labels, label="Select OCR Detection Model",
                                            value=det_models_labels[0])
                    rec_model = gr.Dropdown(rec_models_labels, label="Select OCR Recognition Model",
                                            value=rec_models_labels[0])

                    run_button = gr.Button("Run")
                    gr.Markdown("# Elapsed Time")
                    elapse_text = gr.Text(label="")  # 使用 `gr.Text` 组件展示字符串
            with gr.Column(scale=2):  # 右边列
                # 使用 Markdown 标题分隔各个组件
                gr.Markdown("# Html Render")
                html_output = gr.HTML(label="", elem_classes="scrollable-container")
                gr.Markdown("# Table Boxes")
                table_boxes_output = gr.Image(label="")
                gr.Markdown("# OCR Boxes")
                ocr_boxes_output = gr.Image(label="")

        run_button.click(
            fn=process_image,
            inputs=[img_input, table_engine_type, det_model, rec_model],
            outputs=[html_output, table_boxes_output, ocr_boxes_output, elapse_text]
        )

    demo.launch()


if __name__ == '__main__':
    main()
