import time

import cv2
import gradio as gr
from lineless_table_rec import LinelessTableRecognition
from rapid_table import RapidTable, RapidTableInput
from rapid_table.main import ModelType
from rapidocr_onnxruntime import RapidOCR
from table_cls import TableCls
from wired_table_rec import WiredTableRecognition

from utils import plot_rec_box, LoadImage, format_html, box_4_2_poly_to_box_4_1

import numpy as np

import ocrtest

img_loader = LoadImage()
table_rec_path = "models/table_rec/ch_ppstructure_mobile_v2_SLANet.onnx"
det_model_dir = {
    "mobile_det": "models/ocr/ch_PP-OCRv4_det_server_infer.onnx",
}

rec_model_dir = {
    "mobile_rec": "models/ocr/ch_PP-OCRv4_rec_server_infer.onnx",
}
table_engine_list = [
    "auto",
    "RapidTable(SLANet)",
    "RapidTable(SLANet-plus)",
    "RapidTable(unitable)",
    "wired_table_v2",
    "wired_table_v1",
    "lineless_table"
]

# 示例图片路径
example_images = [
    "images/10.jpg",
    "images/JC.png"
]
rapid_table_engine = RapidTable(RapidTableInput(model_type=ModelType.PPSTRUCTURE_ZH.value, model_path="models/tsr/ch_ppstructure_mobile_v2_SLANet.onnx"))
SLANet_plus_table_Engine = RapidTable(RapidTableInput(model_type=ModelType.SLANETPLUS.value, model_path="models/tsr/slanet-plus.onnx"))
unitable_table_Engine = RapidTable(RapidTableInput(model_type=ModelType.UNITABLE.value, model_path={
            "encoder": f"models/tsr/unitable_encoder.pth",
            "decoder": f"models/tsr/unitable_decoder.pth",
            "vocab": f"models/tsr/unitable_vocab.json",
        }))
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

def trans_char_ocr_res(ocr_res):
    word_result = []
    for res in ocr_res:
        score = res[2]
        for word_box, word in zip(res[3], res[4]):
            word_res = []
            word_res.append(word_box)
            word_res.append(word)
            word_res.append(score)
            word_result.append(word_res)
    return word_result

def select_ocr_model(det_model, rec_model):
    return ocr_engine_dict[f"{det_model}_{rec_model}"]


def select_table_model(img, table_engine_type, det_model, rec_model):
    if table_engine_type == "RapidTable(SLANet)":
        return rapid_table_engine, table_engine_type
    elif table_engine_type == "RapidTable(SLANet-plus)":
        return SLANet_plus_table_Engine, table_engine_type
    elif table_engine_type == "RapidTable(unitable)":
        return unitable_table_Engine, table_engine_type
    elif table_engine_type == "wired_table_v1":
        return wired_table_engine_v1, table_engine_type
    elif table_engine_type == "wired_table_v2":
        print("使用v2 wired table")
        return wired_table_engine_v2, table_engine_type
    elif table_engine_type == "lineless_table":
        return lineless_table_engine, table_engine_type
    elif table_engine_type == "auto":
        cls, elasp = table_cls(img)
        if cls == 'wired':
            table_engine = wired_table_engine_v2
            return table_engine, "wired_table_v2"
        return lineless_table_engine, "lineless_table"


def process_image(img_input, small_box_cut_enhance, table_engine_type, char_ocr, rotated_fix, col_threshold, row_threshold, remove_line, preproc):
    det_model="mobile_det"
    rec_model="mobile_rec"
    img = img_loader(img_input)
    if preproc:
        # img = ocrtest.proc(img)
        pass
    start = time.time()
    table_engine, talbe_type = select_table_model(img, table_engine_type, det_model, rec_model)
    ocr_engine = select_ocr_model(det_model, rec_model)

    ocr_res, ocr_infer_elapse = ocr_engine(img,
                                           max_side_len=2000,
                                        #    det_limit_side_len=960,
                                           det_limit_type="min",
                                        #    det_thresh=0.1,
                                           det_box_thresh=0.1,
                                           text_score=0.01,

                                           return_word_box=char_ocr)
    det_cost, cls_cost, rec_cost = ocr_infer_elapse
    if char_ocr:
        ocr_res = trans_char_ocr_res(ocr_res)
    ocr_boxes = [box_4_2_poly_to_box_4_1(ori_ocr[0]) for ori_ocr in ocr_res]
    if isinstance(table_engine, RapidTable):
        table_results = table_engine(img, ocr_res)
        html, polygons, table_rec_elapse = table_results.pred_html, table_results.cell_bboxes,table_results.elapse
        polygons = [[polygon[0], polygon[1], polygon[4], polygon[5]] for polygon in polygons]
    elif isinstance(table_engine, (WiredTableRecognition, LinelessTableRecognition)):
        html, table_rec_elapse, polygons, logic_points, ocr_res = table_engine(img, ocr_result=ocr_res,
                                                                                   enhance_box_line=small_box_cut_enhance,
                                                                                   rotated_fix=rotated_fix,
                                                                                   col_threshold=col_threshold,
                                                                                   row_threshold=row_threshold)
    # 在获得了 polygons 之后，也就有了表格框线所在的位置
    # 对原图像做处理，去除掉这些框线，只保留内部文字
    # 注意自适应识别框线的宽度
    # Remove table lines while preserving text
    if remove_line:
        # TODO: bad implementation, replicate computation, need improved
        oriimg = img.copy()
        for polygon in polygons:
            # reshape to points
            # print("polygon", polygon)
            points = [[polygon[0], polygon[1]], [polygon[0], polygon[3]], [polygon[2], polygon[3]], [polygon[2], polygon[1]]]
            points = np.array(points, dtype=np.int32)
            def removeline(point1, point2):
                if point1[0] == point2[0]:
                    normal = np.array([1, 0])
                else:
                    normal = np.array([0, 1])
                pa = point1.copy().astype(np.int32)
                pb = point2.copy().astype(np.int32)
                find = False
                for i in range(1, 20):
                    if pa[0] == pb[0]:
                        minv, maxv = sorted([pa[1], pb[1]])
                        avgcolor = np.mean(oriimg[minv:maxv, pa[0], :], axis=0)
                        varcolor = np.var(oriimg[minv:maxv, pa[0], :], axis=0)
                        pa = [pa[0], minv]
                        pb = [pb[0], maxv]
                        # while np.linalg.norm(avgcolor - oriimg[pa[1], pa[0], :]) < 20:
                        #     pa[1] = pa[1] - 1
                        #     avgcolor = np.mean(oriimg[minv:maxv, pa[0], :], axis=0)
                        # while np.linalg.norm(avgcolor - oriimg[pb[1], pb[0], :]) < 20:
                        #     pb[1] = pb[1] + 1
                        #     avgcolor = np.mean(oriimg[minv:maxv, pa[0], :], axis=0)
                    else:
                        minv, maxv = sorted([pa[0], pb[0]])
                        avgcolor = np.mean(oriimg[pa[1], minv:maxv, :], axis=0)
                        varcolor = np.var(oriimg[pa[1], minv:maxv, :], axis=0)
                        pa = [minv, pa[1]]
                        pb = [maxv, pb[1]]
                        # while np.linalg.norm(avgcolor - oriimg[pa[1], pa[0], :]) < 20:
                        #     pa[0] = pa[0] - 1
                        #     avgcolor = np.mean(oriimg[pa[1], minv:maxv, :], axis=0)
                        # while np.linalg.norm(avgcolor - oriimg[pb[1], pb[0], :]) < 20:
                        #     pb[0] = pb[0] + 1
                        #     avgcolor = np.mean(oriimg[pa[1], minv:maxv, :], axis=0)
                    # print("avgcolor", avgcolor)
                    # print(np.max(varcolor))
                    if np.min(avgcolor) < 200:#230 and np.max(varcolor) < 5000:
                        if pa[0] == pb[0]:
                            pa = [pa[0], minv]
                            pb = [pb[0], maxv]
                            while np.linalg.norm(avgcolor - oriimg[pa[1], pa[0], :]) < 20:
                                pa[1] = pa[1] - 1
                                avgcolor = np.mean(oriimg[minv:maxv, pa[0], :], axis=0)
                            while np.linalg.norm(avgcolor - oriimg[pb[1], pb[0], :]) < 20:
                                pb[1] = pb[1] + 1
                                avgcolor = np.mean(oriimg[minv:maxv, pa[0], :], axis=0)
                        else:
                            pa = [minv, pa[1]]
                            pb = [maxv, pb[1]]
                            while np.linalg.norm(avgcolor - oriimg[pa[1], pa[0], :]) < 20:
                                pa[0] = pa[0] - 1
                                avgcolor = np.mean(oriimg[pa[1], minv:maxv, :], axis=0)
                            while np.linalg.norm(avgcolor - oriimg[pb[1], pb[0], :]) < 20:
                                pb[0] = pb[0] + 1
                                avgcolor = np.mean(oriimg[pa[1], minv:maxv, :], axis=0)
                        cv2.line(img, tuple(pa), tuple(pb), (255, 255, 255), 1)
                        find = True
                    elif find:
                        break
            
                    pa = pa + normal
                    pb = pb + normal
                pa = point1.copy() - normal
                pb = point2.copy() - normal
                find  = False
                for i in range(1, 20):
                    if pa[0] == pb[0]:
                        minv, maxv = sorted([pa[1], pb[1]])
                        avgcolor = np.mean(oriimg[minv:maxv, pa[0], :], axis=0)
                        varcolor = np.var(oriimg[minv:maxv, pa[0], :], axis=0)
                        pa = [pa[0], minv]
                        pb = [pb[0], maxv]
                        # while np.linalg.norm(avgcolor - oriimg[pa[1], pa[0], :]) < 20:
                        #     pa[1] = pa[1] - 1
                        #     avgcolor = np.mean(oriimg[minv:maxv, pa[0], :], axis=0)
                        # while np.linalg.norm(avgcolor - oriimg[pb[1], pb[0], :]) < 20:
                        #     pb[1] = pb[1] + 1
                        #     avgcolor = np.mean(oriimg[minv:maxv, pa[0], :], axis=0)
                    else:
                        minv, maxv = sorted([pa[0], pb[0]])
                        avgcolor = np.mean(oriimg[pa[1], minv:maxv, :], axis=0)
                        varcolor = np.var(oriimg[pa[1], minv:maxv, :], axis=0)
                        pa = [minv, pa[1]]
                        pb = [maxv, pb[1]]
                        # while np.linalg.norm(avgcolor - oriimg[pa[1], pa[0], :]) < 20:
                        #     pa[0] = pa[0] - 1
                        #     avgcolor = np.mean(oriimg[pa[1], minv:maxv, :], axis=0)
                        # while np.linalg.norm(avgcolor - oriimg[pb[1], pb[0], :]) < 20:
                        #     pb[0] = pb[0] + 1
                        #     avgcolor = np.mean(oriimg[pa[1], minv:maxv, :], axis=0)
                    # print("avgcolor", avgcolor)
                    if np.min(avgcolor) < 200:#230 and np.max(varcolor) < 5000:
                        if pa[0] == pb[0]:
                            pa = [pa[0], minv]
                            pb = [pb[0], maxv]
                            while np.linalg.norm(avgcolor - oriimg[pa[1], pa[0], :]) < 20:
                                pa[1] = pa[1] - 1
                                avgcolor = np.mean(oriimg[minv:maxv, pa[0], :], axis=0)
                            while np.linalg.norm(avgcolor - oriimg[pb[1], pb[0], :]) < 20:
                                pb[1] = pb[1] + 1
                                avgcolor = np.mean(oriimg[minv:maxv, pa[0], :], axis=0)
                        else:
                            pa = [minv, pa[1]]
                            pb = [maxv, pb[1]]
                            while np.linalg.norm(avgcolor - oriimg[pa[1], pa[0], :]) < 20:
                                pa[0] = pa[0] - 1
                                avgcolor = np.mean(oriimg[pa[1], minv:maxv, :], axis=0)
                            while np.linalg.norm(avgcolor - oriimg[pb[1], pb[0], :]) < 20:
                                pb[0] = pb[0] + 1
                                avgcolor = np.mean(oriimg[pa[1], minv:maxv, :], axis=0)
                        cv2.line(img, tuple(pa), tuple(pb), (255, 255, 255), 1)
                        find = True
                    elif find:
                        break
                    pa = pa - normal
                    pb = pb - normal
            removeline(points[0], points[1])
            removeline(points[1], points[2])
            removeline(points[2], points[3])
            removeline(points[3], points[0])
        cv2.imwrite("clean_img.jpg", img)

        # re ocr 
        ocr_res, ocr_infer_elapse = ocr_engine(img,
                                               max_side_len=2000,
                                               det_limit_type="min",
                                               det_box_thresh=0.1,
                                               text_score=0.01,
                                               return_word_box=char_ocr)
        det_cost, cls_cost, rec_cost = ocr_infer_elapse
        if char_ocr:
            ocr_res = trans_char_ocr_res(ocr_res)
        ocr_boxes = [box_4_2_poly_to_box_4_1(ori_ocr[0]) for ori_ocr in ocr_res]
        html, table_rec_elapse, polygons, logic_points, ocr_res = table_engine(oriimg, ocr_result=ocr_res,
                                                                                   enhance_box_line=small_box_cut_enhance,
                                                                                   rotated_fix=rotated_fix,
                                                                                   col_threshold=col_threshold,
                                                                                   row_threshold=row_threshold)
        
    
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
            "<h1 style='text-align: center;'><a href='https://github.com/RapidAI/TableStructureRec?tab=readme-ov-file'>TableStructureRec</a> & <a href='https://github.com/RapidAI/RapidTable'>RapidTable</a></h1>"
        )
        gr.HTML('''
                                        <div class="header-links">
                                          <a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.12-aff.svg"></a>
                                          <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Mac%2C%20Win-pink.svg"></a>
                                          <a href="https://pypi.org/project/lineless-table-rec/"><img alt="PyPI" src="https://img.shields.io/pypi/v/lineless-table-rec"></a>
                                          <a href="https://pepy.tech/project/lineless-table-rec"><img src="https://static.pepy.tech/personalized-badge/lineless-table-rec?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20Lineless"></a>
                                          <a href="https://pepy.tech/project/wired-table-rec"><img src="https://static.pepy.tech/personalized-badge/wired-table-rec?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20Wired"></a>
                                          <a href="https://pepy.tech/project/rapid-table"><img src="https://static.pepy.tech/personalized-badge/rapid-table?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20RapidTable"></a>
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
                        label="识别框切割增强(wiredV2,关闭避免多余切割，开启减少漏切割)",
                        value=True
                    )
                    remove_line = gr.Checkbox(
                        label="去除表格线(hzw)",
                        value=True
                    )
                    preproc = gr.Checkbox(
                        label="预处理(去噪、二值化等，效果不佳)(hzw)",
                        value=False
                    )
                    char_ocr = gr.Checkbox(
                        label="单字符OCR匹配",
                        value=True
                    )
                    rotate_adapt = gr.Checkbox(
                        label="表格旋转识别增强(wiredV2)",
                        value=False
                    )
                    col_threshold = gr.Slider(
                        label="同列x坐标距离阈值(wiredV2)",
                        minimum=5,
                        maximum=100,
                        value=5,
                        step=5
                    )
                    row_threshold = gr.Slider(
                        label="同行y坐标距离阈值(wiredV2)",
                        minimum=5,
                        maximum=100,
                        value=5,
                        step=5
                    )

                    # det_model = gr.Dropdown(det_models_labels, label="Select OCR Detection Model",
                    #                         value=det_models_labels[0])
                    # rec_model = gr.Dropdown(rec_models_labels, label="Select OCR Recognition Model",
                    #                         value=rec_models_labels[0])

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
            inputs=[img_input, small_box_cut_enhance, table_engine_type, char_ocr, rotate_adapt, col_threshold, row_threshold, remove_line, preproc],
            outputs=[html_output, table_boxes_output, ocr_boxes_output, elapse_text]
        )

    demo.launch()


if __name__ == '__main__':
    main()
