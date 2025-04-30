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
import cv2
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt

def detect_text_gray_range(img):
    # 1. 读取并转为灰度图
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. reshape 为一列数据，用于GMM建模
    pixels = img.reshape(-1, 1)

    # 3. 使用GMM拟合两个高斯分布
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(pixels)
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()
    weights = gmm.weights_

    # 4. 将分布比例较大的作为背景，较小的作为文字
    text_class = 0 if weights[0] < weights[1] else 1
    text_mean = means[text_class]
    text_std = stds[text_class]
    background_class = 1 - text_class
    background_mean = means[background_class]
    background_std = stds[background_class]

    # 5. 计算文字颜色范围
    # 由于文字颜色方差较大，我们计算背景颜色，取补集
    if background_mean > 128:
        bg_std_ratio = (255 - background_mean) / background_std
        lower = 0
        upper = int(background_mean * 0.5 + text_mean * 0.5)
        # upper = int(background_mean - bg_std_ratio * background_std)
    else:
        bg_std_ratio = background_mean / background_std
        # lower = int(background_mean + bg_std_ratio * background_std)
        lower = int(background_mean * 0.5 + text_mean * 0.5)
        upper = 255
    # print(text_mean, text_std)

    # 6.计算这个范围内像素的占比
    mask = cv2.inRange(img, lower, upper)
    pixel_count = np.sum(mask > 0)
    total_count = img.size
    text_ratio = pixel_count / total_count

    print(weights, means, stds)
    print("Text mean:", text_mean)
    print("Text color range:", lower, upper)
    print("Text color ratio:", text_ratio)

    # 绘制直方图和GMM拟合结果，保存到文件
    plt.hist(pixels, bins=256, range=(0, 255), density=True, alpha=0.5)
    x = np.linspace(0, 255, 256).reshape(-1, 1)
    logprob = gmm.score_samples(x)
    pdf = np.exp(logprob)
    plt.plot(x, pdf, '-k', label='GMM')
    # 分别绘制两个正态分布
    for i in range(2):
        mean = means[i]
        std = stds[i]
        weight = weights[i]
        plt.plot(x, weight * np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi)), label=f'Gaussian {i+1}')
    plt.title('GMM Fit')
    plt.xlabel('Pixel Value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig("gmm_fit.png")

    return lower, upper, text_ratio

def compute_ratio(img, lower, upper):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 计算文字颜色范围
    mask = cv2.inRange(img, lower, upper)
    pixel_count = np.sum(mask > 0)
    total_count = img.size
    text_ratio = pixel_count / total_count
    # print(text_ratio)
    return text_ratio

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
            avg_text_height = np.mean(
                [polygon[5] - polygon[1] for polygon in ocr_polygons["polygons"]]
            )
            cell_box_det_map = {}
            ocr_cell_polys = {}
            ocr_results = []
            # 框线过滤
            def is_line(p1x, p1y, p2x, p2y):
                # print(p1x, p1y, p2x, p2y)
                if p1x < 0 or p1y < 0 or p2x < 0 or p2y < 0 or p1x >= img.shape[1] or p1y >= img.shape[0] or p2x >= img.shape[1] or p2y >= img.shape[0]:
                    return False
                if p1x == p2x:
                    line = img[p1y:p2y, p1x]
                elif p1y == p2y:
                    line = img[p1y, p1x:p2x]
                color = np.mean(line)
                if color < 200:
                    return True
                else:
                    return False
            for j, polygon in enumerate(polygons):
                find = False
                x1, y1, x2, y2 = polygon[0][0], polygon[0][1], polygon[2][0], polygon[2][1]
                if x1 == x2 or y1 == y2:
                    continue
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
                while is_line(x1, y1, x1, y2):
                    x1 += 1
                while is_line(x2, y1, x2, y2):
                    x2 -= 1
                while is_line(x1, y2, x2, y2):
                    y2 -= 1
                while is_line(x1, y1, x2, y1):
                    y1 += 1
                polygons[j] = [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                ]
                
                for ocr_polygon in ocr_polygons['polygons']:
                    # 转换为统一的格式
                    x1_ocr, y1_ocr, x2_ocr, y2_ocr = ocr_polygon[0], ocr_polygon[1], ocr_polygon[4], ocr_polygon[5]
                    if x1_ocr == x2_ocr or y1_ocr == y2_ocr:
                        continue
                    if x1_ocr > x2_ocr:
                        x1_ocr, x2_ocr = x2_ocr, x1_ocr
                    if y1_ocr > y2_ocr:
                        y1_ocr, y2_ocr = y2_ocr, y1_ocr

                    # 求交集
                    x1_inter = int(max(x1, x1_ocr))
                    y1_inter = int(max(y1, y1_ocr))
                    x2_inter = int(min(x2, x2_ocr))
                    y2_inter = int(min(y2, y2_ocr))
                    # print(x1_inter, y1_inter, x2_inter, y2_inter)
                    if x1_inter >= x2_inter or y1_inter >= y2_inter:
                        continue
                    # find
                    find = True
                    if j not in ocr_cell_polys:
                        ocr_cell_polys[j] = [[x1_inter, y1_inter, x2_inter, y2_inter]]
                    else:
                        ocr_cell_polys[j].append([x1_inter, y1_inter, x2_inter, y2_inter])
                    # ocr_result = modeltest.ocr_recognition(img[y1_inter:y2_inter+1, x1_inter:x2_inter+1])['text'][0].strip()
                    # if ocr_result != "":
                    #     find = True
                    #     gt_box = [[
                    #         [x1_inter, y1_inter],
                    #         [x2_inter, y1_inter],
                    #         [x2_inter, y2_inter],
                    #         [x1_inter, y2_inter], 
                    #     ], ocr_result, 1.0]
                    #     if j not in cell_box_det_map:
                    #         cell_box_det_map[j] = [gt_box]
                    #     else:
                    #         cell_box_det_map[j].append(gt_box)
                    #     ocr_results.append(gt_box)
                # if find:
                #     heights = [y2 - y1 for x1, y1, x2, y2 in polys[j]]
                #     pass
                # else:
                #     ocr_result = modeltest.ocr_recognition(img[y1:y2+1, x1:x2+1])['text'][0].strip()
                #     if ocr_result != "":
                #         gt_box = [[
                #             [x1, y1],
                #             [x2, y1],
                #             [x2, y2],
                #             [x1, y2], 
                #         ], ocr_result, 1.0]
                #         cell_box_det_map[j] = [gt_box]
                #         ocr_results.append(gt_box)
            img_blur = cv2.GaussianBlur(img.copy(), (15, 15), 0)
            cv2.imwrite("img_blur.png", img_blur)
            color_min, color_max, text_ratio = detect_text_gray_range(img_blur)
            for j, polygon in enumerate(polygons):
                findtext = False
                while True: # 非真循环，只在没有找到文本时重试一次
                    if ocr_cell_polys.get(j) is None:
                        # 没有文字
                        # TODO
                        print("no text")
                        ocr_cell_polys[j] = [[polygon[0][0], int(polygon[0][1] * 0.5 + polygon[2][1] * 0.5 - avg_text_height / 2), polygon[2][0], int(polygon[0][1] * 0.5 + polygon[2][1] * 0.5 + avg_text_height / 2)]]
                    
                    if True:
                        x1_cell, y1_cell, x2_cell, y2_cell = polygon[0][0], polygon[0][1], polygon[2][0], polygon[2][1]
                        # 根据y1排序
                        ocr_cell_polys[j] = sorted(ocr_cell_polys[j], key=lambda x: x[1])
                        for i in range(1, len(ocr_cell_polys[j])):
                            poly = ocr_cell_polys[j][i]
                            poly_last = ocr_cell_polys[j][i-1]
                            x1, y1, x2, y2 = poly[0], poly[1], poly[2], poly[3]
                            x1_last, y1_last, x2_last, y2_last = poly_last[0], poly_last[1], poly_last[2], poly_last[3]
                            if y1 > y2_last:
                                mid = int((y1 + y2_last) / 2)
                                ocr_cell_polys[j][i-1] = [x1_last, y1_last, x2_last, mid]
                                ocr_cell_polys[j][i] = [x1, mid, x2, y2]
                        for k in range(0, len(ocr_cell_polys[j])):
                            # 适当拓展
                            ocr_cell_poly = ocr_cell_polys[j][k]
                            x1, y1, x2, y2 = ocr_cell_poly[0], ocr_cell_poly[1], ocr_cell_poly[2], ocr_cell_poly[3]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            if x1 >= x2 or y1 >= y2:
                                continue
                            text_ratio = compute_ratio(img_blur[y1:y2+1, x1:x2+1], color_min, color_max)
                            edge_ratio = 0.3
                            similarx = [poly for poly in ocr_cell_polys[j] 
                                        if (x1 <= poly[0] <= x2 or x1 <= poly[2] <= x2 or poly[0] <= x1 <= poly[2] or poly[0] <= x2 <= poly[2])
                                        and (poly[1] > y1 or poly[3] < y2)]
                            toppoly = [poly[3] for poly in similarx if poly[3] < y2]
                            botpoly = [poly[1] for poly in similarx if poly[1] > y1]
                            maxtop = max(toppoly) if len(toppoly) > 0 else y1_cell
                            maxbot = min(botpoly) if len(botpoly) > 0 else y2_cell
                            # print(maxtop, maxbot, y1, y2)
                            max_extend = int(max(min(max(avg_text_height - (y2 - y1), 10) * 2, maxbot - y2),0))
                            skipchance = int(max_extend / 3)
                            lasti = 1
                            for i in range(1, max_extend):
                                if y2 + i > y2_cell or i == max_extend - 1:
                                    y2 = y2 + lasti - 1
                                    break
                                if compute_ratio(img_blur[y2:y2+i+1, x1:x2+1], color_min, color_max) < text_ratio * edge_ratio:
                                    skipchance -= 1
                                    if skipchance == 0:
                                        if lasti != 1:print(lasti)
                                        y2 = y2 + lasti - 1
                                        break
                                else: 
                                    lasti = i
                            max_extend = int(max(min(max(avg_text_height - (y2 - y1), 10) * 2, y1 - maxtop), 0))
                            skipchance = int(max_extend / 2)
                            lasti = 1
                            for i in range(1, max_extend):
                                if y1 - i < y1_cell or i == max_extend - 1:
                                    y1 = y1 - lasti + 1
                                    break
                                if compute_ratio(img_blur[y1-i:y1+1, x1:x2+1], color_min, color_max) < text_ratio * edge_ratio:
                                    skipchance -= 1
                                    if skipchance == 0:
                                        y1 = y1 - lasti + 1
                                        break
                                else:
                                    lasti = i
                            midx = int((x1 + x2) / 2)
                            bg_ratio = 0.1
                            # while x1 < midx - avg_text_height:
                            #     ratio = compute_ratio(img_blur[y1:y2+1, x1:x1+1], color_min, color_max)
                            #     if ratio < text_ratio * bg_ratio or ratio > 0.95:
                            #         x1 += 1
                            #         print("add")
                            #     else:
                            #         break
                            # while x2 > midx + avg_text_height:
                            #     ratio = compute_ratio(img_blur[y1:y2+1, x2:x2+1], color_min, color_max)
                            #     if ratio < text_ratio * bg_ratio or ratio > 0.95:
                            #         x2 -= 1
                            #     else:
                            #         break
                            # print(modeltest.ocr_recognition(img[y1:y2+2, x1:x2+2])['text'][0].strip())
                            ocr_cell_polys[j][k] = [x1, y1, x2, y2]
                    for ocr_cell_poly in ocr_cell_polys[j]:
                        x1, y1, x2, y2 = ocr_cell_poly[0], ocr_cell_poly[1], ocr_cell_poly[2], ocr_cell_poly[3]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        if x1 >= x2 or y1 >= y2 or y2 - y1 < avg_text_height * 0.2:
                            ocr_result = ""
                        else: 
                            try:
                                ocr_result = modeltest.ocr_recognition(img[y1:y2+2, x1:x2+2])['text'][0].strip()
                            except:
                                ocr_result = ""
                        if ocr_result != "":
                            findtext = True
                            gt_box = [[
                                [ocr_cell_poly[0], ocr_cell_poly[1]],
                                [ocr_cell_poly[2], ocr_cell_poly[1]],
                                [ocr_cell_poly[2], ocr_cell_poly[3]],
                                [ocr_cell_poly[0], ocr_cell_poly[3]], 
                            ], ocr_result, 1.0]
                            if j not in cell_box_det_map:
                                cell_box_det_map[j] = [gt_box]
                            else:
                                cell_box_det_map[j].append(gt_box)
                            ocr_results.append(gt_box)
                    if findtext:
                        break
                    else:
                        findtext = True #只重试一次
                        ocr_cell_polys[j] = None

                    

                    

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