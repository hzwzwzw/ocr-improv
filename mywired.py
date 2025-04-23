from wired_table_rec import WiredTableRecognition
from wired_table_rec.main import (
    plot_html_table,
    sorted_ocr_boxes,
    box_4_2_poly_to_box_4_1,
)
import traceback
import time
import logging
import modeltest
import numpy as np
# import cv2
class mywired(WiredTableRecognition):
    def __call__(self, img, ocr_result = None, **kwargs):

        s = time.perf_counter()
        rec_again = True
        need_ocr = True
        col_threshold = 15
        row_threshold = 10
        if kwargs:
            rec_again = kwargs.get("rec_again", True)
            need_ocr = kwargs.get("need_ocr", True)
            col_threshold = kwargs.get("col_threshold", 15)
            row_threshold = kwargs.get("row_threshold", 10)
        img = self.load_img(img)
        polygons, rotated_polygons = self.table_line_rec(img, **kwargs)
        if polygons is None:
            logging.warning("polygons is None.")
            return "", 0.0, None, None, None

        try:
            table_res, logi_points = self.table_recover(
                rotated_polygons, row_threshold, col_threshold
            )
            # 将坐标由逆时针转为顺时针方向，后续处理与无线表格对齐
            polygons[:, 1, :], polygons[:, 3, :] = (
                polygons[:, 3, :].copy(),
                polygons[:, 1, :].copy(),
            )
            # 修改开始
            ocr_polygons = modeltest.ocr_detection(img)
            cell_box_det_map = {}
            ocr_results = []
            for j, polygon in enumerate(polygons):
                find = False
                for ocr_polygon in ocr_polygons['polygons']:
                    # 转换为统一的格式
                    x1, y1, x2, y2 = polygon[0][0], polygon[0][1], polygon[2][0], polygon[2][1]
                    x1_ocr, y1_ocr, x2_ocr, y2_ocr = ocr_polygon[0], ocr_polygon[1], ocr_polygon[4], ocr_polygon[5]
                    if x1 == x2 or y1 == y2:
                        continue
                    if x1 > x2:
                        x1, x2 = x2, x1
                    if y1 > y2:
                        y1, y2 = y2, y1
                    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
                    if x1_ocr == x2_ocr or y1_ocr == y2_ocr:
                        continue
                    if x1_ocr > x2_ocr:
                        x1_ocr, x2_ocr = x2_ocr, x1_ocr
                    if y1_ocr > y2_ocr:
                        y1_ocr, y2_ocr = y2_ocr, y1_ocr
                    # 框线过滤
                    def is_line(p1x, p1y, p2x, p2y):
                        print(p1x, p1y, p2x, p2y)
                        if p1x == p2x:
                            line = img[p1y:p2y, p1x]
                        elif p1y == p2y:
                            line = img[p1y, p1x:p2x]
                        color = np.mean(line)
                        if color < 200:
                            return True
                        else:
                            return False
                    while is_line(x1, y1, x1, y2):
                        x1 += 1
                    while is_line(x2, y1, x2, y2):
                        x2 -= 1
                    while is_line(x1, y2, x2, y2):
                        y2 -= 1
                    while is_line(x1, y1, x2, y1):
                        y1 += 1

                    # 求交集
                    x1_inter = int(max(x1, x1_ocr))
                    y1_inter = int(max(y1, y1_ocr))
                    x2_inter = int(min(x2, x2_ocr))
                    y2_inter = int(min(y2, y2_ocr))
                    # print(x1_inter, y1_inter, x2_inter, y2_inter)
                    if x1_inter >= x2_inter or y1_inter >= y2_inter:
                        continue
                    ocr_result = modeltest.ocr_recognition(img[y1_inter:y2_inter+1, x1_inter:x2_inter+1])['text'][0].strip()
                    if ocr_result != "":
                        find = True
                        gt_box = [[
                            [x1_inter, y1_inter],
                            [x2_inter, y1_inter],
                            [x2_inter, y2_inter],
                            [x1_inter, y2_inter], 
                        ], ocr_result, 1.0]
                        if j not in cell_box_det_map:
                            cell_box_det_map[j] = [gt_box]
                        else:
                            cell_box_det_map[j].append(gt_box)
                        ocr_results.append(gt_box)
                if not find:
                    ocr_result = modeltest.ocr_recognition(img[y1:y2+1, x1:x2+1])['text'][0].strip()
                    if ocr_result != "":
                        gt_box = [[
                            [x1, y1],
                            [x2, y1],
                            [x2, y2],
                            [x1, y2], 
                        ], ocr_result, 1.0]
                        cell_box_det_map[j] = [gt_box]
                        ocr_results.append(gt_box)
            # 修改结束
            # print(ocr_results[0])
            # cell_box_det_map = self.re_rec(img, polygons, cell_box_det_map, rec_again)
            # 转换为中间格式，修正识别框坐标,将物理识别框，逻辑识别框，ocr识别框整合为dict，方便后续处理
            t_rec_ocr_list = self.transform_res(cell_box_det_map, polygons, logi_points)
            # 将每个单元格中的ocr识别结果排序和同行合并，输出的html能完整保留文字的换行格式
            t_rec_ocr_list = self.sort_and_gather_ocr_res(t_rec_ocr_list)
            # cell_box_map =
            logi_points = [t_box_ocr["t_logic_box"] for t_box_ocr in t_rec_ocr_list]
            cell_box_det_map = {
                i: [ocr_box_and_text[1] for ocr_box_and_text in t_box_ocr["t_ocr_res"]]
                for i, t_box_ocr in enumerate(t_rec_ocr_list)
            }
            table_str = plot_html_table(logi_points, cell_box_det_map)
            ocr_boxes_res = [
                box_4_2_poly_to_box_4_1(ori_ocr[0]) for ori_ocr in ocr_results
            ]
            sorted_ocr_boxes_res, _ = sorted_ocr_boxes(ocr_boxes_res)
            sorted_polygons = [box_4_2_poly_to_box_4_1(box) for box in polygons]
            sorted_logi_points = logi_points
            table_elapse = time.perf_counter() - s

        except Exception:
            logging.warning(traceback.format_exc())
            return "", 0.0, None, None, None
        return (
            table_str,
            table_elapse,
            sorted_polygons,
            sorted_logi_points,
            sorted_ocr_boxes_res,
        )