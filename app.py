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
table_cls = TableCls()
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


def process_image(img, table_engine_type, det_model, rec_model, small_box_cut_enhance):
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
            if not small_box_cut_enhance:
                html, table_rec_elapse, polygons, logic_points, ocr_res = table_engine(
                    img, ocr_result=ocr_res,
                    morph_close=False, more_h_lines=False, more_v_lines=False, extend_line=False
                )
            else:
                html, table_rec_elapse, polygons, logic_points, ocr_res = table_engine(
                    img, ocr_result=ocr_res
                )

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
        .header-links {
            text-align: center;
        }
        .header-links a {
            display: inline-block;
            text-align: center;
            margin-right: 10px;  /* 调整间距 */
        }
    """) as demo:
        gr.HTML(
            "<h1 style='text-align: center;'><a href='https://github.com/RapidAI/TableStructureRec?tab=readme-ov-file'>TableStructureRec</a></h1>"
        )
        gr.HTML('''
                                        <div class="header-links">
                                          <a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.12-aff.svg"></a>
                                          <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Mac%2C%20Win-pink.svg"></a>
                                          <a href="https://pypi.org/project/lineless-table-rec/"><img alt="PyPI" src="https://img.shields.io/pypi/v/lineless-table-rec"></a>
                                          <a href="https://pepy.tech/project/lineless-table-rec"><img src="https://static.pepy.tech/personalized-badge/lineless-table-rec?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20Lineless"></a>
                                          <a href="https://pepy.tech/project/wired-table-rec"><img src="https://static.pepy.tech/personalized-badge/wired-table-rec?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20Wired"></a>
                                          <a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
                                          <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
                                          <a href="https://github.com/RapidAI/TableStructureRec/blob/c41bbd23898cb27a957ed962b0ffee3c74dfeff1/LICENSE"><img alt="GitHub" src="https://img.shields.io/badge/license-Apache 2.0-blue"></a>
                                        </div>
                                        ''')
        with gr.Row():  # 两列布局
            with gr.Tab("Options"):
                with gr.Column(variant="panel", scale=1):  # 侧边栏，宽度比例为1
                    img_input = gr.Image(label="Upload or Select Image", sources="upload", value="images/lineless3.jpg")

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
                    small_box_cut_enhance = gr.Checkbox(
                        label="识别框切割增强(关闭避免多余切割，开启减少漏切割)",
                        value=True
                    )
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
            inputs=[img_input, table_engine_type, det_model, rec_model, small_box_cut_enhance],
            outputs=[html_output, table_boxes_output, ocr_boxes_output, elapse_text]
        )

    demo.launch()


if __name__ == '__main__':
    main()

html, elasp, polygons, logic_points, ocr_res = table_engine(
    img,
    morph_close=True,  # 是否进行形态学操作,辅助找到更多线框,默认为True
    more_h_lines=True,  # 是否基于线框检测结果进行更多水平线检查，辅助找到更小线框, 默认为True
    h_lines_threshold=100,  # 必须开启more_h_lines, 连接横线检测像素阈值，小于该值会生成新横线，默认为100
    more_v_lines=True,  # 是否基于线框检测结果进行更多垂直线检查，辅助找到更小线框, 默认为True
    v_lines_threshold=15,  # 必须开启more_v_lines, 连接竖线检测像素阈值，小于该值会生成新竖线，默认为15
    extend_line=True,  # 是否基于线框检测结果进行线段延长，辅助找到更多线框, 默认为True
    need_ocr=True,  # 是否进行OCR识别, 默认为True
    rec_again=True,  # 是否针对未识别到文字的表格框,进行单独截取再识别,默认为True
)
