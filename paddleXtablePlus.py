# import time
# from io import BytesIO
# from pathlib import Path
# from typing import Union, List, Tuple
#
# import cv2
# from PIL import Image, UnidentifiedImageError
#
# import numpy as np
# from paddle.inference import Config, create_predictor
# InputType = Union[str, np.ndarray, bytes, Path, Image.Image]
# # paddle2onnx --model_dir C:\Users\51954\.paddlex\official_models\SLANet_plus --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./table.onnx --opset_version 16 --enable_onnx_checker
# # paddle2onnx --model_dir ./ --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./table.onnx --opset_version 16 --enable_onnx_checker
#
# def get_boxes_recs(
#         ocr_result: List[Union[List[List[float]], str, str]], h: int, w: int
# ) -> Tuple[np.ndarray, Tuple[str, str]]:
#     dt_boxes, rec_res, scores = list(zip(*ocr_result))
#     rec_res = list(zip(rec_res, scores))
#
#     r_boxes = []
#     for box in dt_boxes:
#         box = np.array(box)
#         x_min = max(0, box[:, 0].min() - 1)
#         x_max = min(w, box[:, 0].max() + 1)
#         y_min = max(0, box[:, 1].min() - 1)
#         y_max = min(h, box[:, 1].max() + 1)
#         box = [x_min, y_min, x_max, y_max]
#         r_boxes.append(box)
#     dt_boxes = np.array(r_boxes)
#     return dt_boxes, rec_res
# def distance(box_1, box_2):
#     x1, y1, x2, y2 = box_1
#     x3, y3, x4, y4 = box_2
#     dis = abs(x3 - x1) + abs(y3 - y1) + abs(x4 - x2) + abs(y4 - y2)
#     dis_2 = abs(x3 - x1) + abs(y3 - y1)
#     dis_3 = abs(x4 - x2) + abs(y4 - y2)
#     return dis + min(dis_2, dis_3)
#
# def convert_corners_to_bounding_boxes(corners):
#     """
#     转换给定的角点坐标到边界框坐标 (xmin, ymin, xmax, ymax)。
#
#     参数:
#     corners : numpy.ndarray
#         形状为 (n, 8) 的数组，每行包含四个角点的坐标 (x1, y1, x2, y2, x3, y3, x4, y4)。
#
#     返回:
#     bounding_boxes : numpy.ndarray
#         形状为 (n, 4) 的数组，每行包含 (xmin, ymin, xmax, ymax)。
#     """
#     # 分别提取四个角点的 x 和 y 坐标
#     x1, y1, x2, y2, x3, y3, x4, y4 = np.split(corners, 8, axis=1)
#
#     # 计算 xmin, ymin, xmax, ymax
#     xmin = np.min(np.hstack((x1, x2, x3, x4)), axis=1, keepdims=True)
#     ymin = np.min(np.hstack((y1, y2, y3, y4)), axis=1, keepdims=True)
#     xmax = np.max(np.hstack((x1, x2, x3, x4)), axis=1, keepdims=True)
#     ymax = np.max(np.hstack((y1, y2, y3, y4)), axis=1, keepdims=True)
#
#     # 拼接成新的数组
#     bounding_boxes = np.concatenate((xmin, ymin, xmax, ymax), axis=1)
#
#     return bounding_boxes
# def compute_iou(rec1, rec2):
#     """
#     computing IoU
#     :param rec1: (y0, x0, y1, x1), which reflects
#             (top, left, bottom, right)
#     :param rec2: (y0, x0, y1, x1)
#     :return: scala value of IoU
#     """
#     # computing area of each rectangles
#     S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
#     S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
#
#     # computing the sum_area
#     sum_area = S_rec1 + S_rec2
#
#     # find the each edge of intersect rectangle
#     left_line = max(rec1[1], rec2[1])
#     right_line = min(rec1[3], rec2[3])
#     top_line = max(rec1[0], rec2[0])
#     bottom_line = min(rec1[2], rec2[2])
#
#     # judge if there is an intersect
#     if left_line >= right_line or top_line >= bottom_line:
#         return 0.0
#     else:
#         intersect = (right_line - left_line) * (bottom_line - top_line)
#         return (intersect / (sum_area - intersect)) * 1.0
#
# class LoadImageError(Exception):
#     pass
#
#
# class LoadImage:
#     def __init__(self):
#         pass
#
#     def __call__(self, img: InputType) -> np.ndarray:
#         if not isinstance(img, InputType.__args__):
#             raise LoadImageError(
#                 f"The img type {type(img)} does not in {InputType.__args__}"
#             )
#
#         origin_img_type = type(img)
#         img = self.load_img(img)
#         img = self.convert_img(img, origin_img_type)
#         return img
#
#     def load_img(self, img: InputType) -> np.ndarray:
#         if isinstance(img, (str, Path)):
#             self.verify_exist(img)
#             try:
#                 img = np.array(Image.open(img))
#             except UnidentifiedImageError as e:
#                 raise LoadImageError(f"cannot identify image file {img}") from e
#             return img
#
#         if isinstance(img, bytes):
#             img = np.array(Image.open(BytesIO(img)))
#             return img
#
#         if isinstance(img, np.ndarray):
#             return img
#
#         if isinstance(img, Image.Image):
#             return np.array(img)
#
#         raise LoadImageError(f"{type(img)} is not supported!")
#
#     def convert_img(self, img: np.ndarray, origin_img_type):
#         if img.ndim == 2:
#             return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#
#         if img.ndim == 3:
#             channel = img.shape[2]
#             if channel == 1:
#                 return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#
#             if channel == 2:
#                 return self.cvt_two_to_three(img)
#
#             if channel == 3:
#                 if issubclass(origin_img_type, (str, Path, bytes, Image.Image)):
#                     return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#                 return img
#
#             if channel == 4:
#                 return self.cvt_four_to_three(img)
#
#             raise LoadImageError(
#                 f"The channel({channel}) of the img is not in [1, 2, 3, 4]"
#             )
#
#         raise LoadImageError(f"The ndim({img.ndim}) of the img is not in [2, 3]")
#
#     @staticmethod
#     def cvt_two_to_three(img: np.ndarray) -> np.ndarray:
#         """gray + alpha → BGR"""
#         img_gray = img[..., 0]
#         img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
#
#         img_alpha = img[..., 1]
#         not_a = cv2.bitwise_not(img_alpha)
#         not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)
#
#         new_img = cv2.bitwise_and(img_bgr, img_bgr, mask=img_alpha)
#         new_img = cv2.add(new_img, not_a)
#         return new_img
#
#     @staticmethod
#     def cvt_four_to_three(img: np.ndarray) -> np.ndarray:
#         """RGBA → BGR"""
#         r, g, b, a = cv2.split(img)
#         new_img = cv2.merge((b, g, r))
#
#         not_a = cv2.bitwise_not(a)
#         not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)
#
#         new_img = cv2.bitwise_and(new_img, new_img, mask=a)
#         new_img = cv2.add(new_img, not_a)
#         return new_img
#
#     @staticmethod
#     def verify_exist(file_path: Union[str, Path]):
#         if not Path(file_path).exists():
#             raise LoadImageError(f"{file_path} does not exist.")
#
#
# class TableMatch:
#     def __init__(self, filter_ocr_result=True, use_master=False):
#         self.filter_ocr_result = filter_ocr_result
#         self.use_master = use_master
#
#     def __call__(self, pred_structures, pred_bboxes, dt_boxes, rec_res):
#         if self.filter_ocr_result:
#             dt_boxes, rec_res = self._filter_ocr_result(pred_bboxes, dt_boxes, rec_res)
#         matched_index = self.match_result(dt_boxes, pred_bboxes)
#         pred_html, pred = self.get_pred_html(pred_structures, matched_index, rec_res)
#         return pred_html
#
#     def match_result(self, dt_boxes, pred_bboxes):
#         matched = {}
#         for i, gt_box in enumerate(dt_boxes):
#             distances = []
#             for j, pred_box in enumerate(pred_bboxes):
#                 if len(pred_box) == 8:
#                     pred_box = [
#                         np.min(pred_box[0::2]),
#                         np.min(pred_box[1::2]),
#                         np.max(pred_box[0::2]),
#                         np.max(pred_box[1::2]),
#                     ]
#                 distances.append(
#                     (distance(gt_box, pred_box), 1.0 - compute_iou(gt_box, pred_box))
#                 )  # compute iou and l1 distance
#             sorted_distances = distances.copy()
#             # select det box by iou and l1 distance
#             sorted_distances = sorted(
#                 sorted_distances, key=lambda item: (item[1], item[0])
#             )
#             if distances.index(sorted_distances[0]) not in matched.keys():
#                 matched[distances.index(sorted_distances[0])] = [i]
#             else:
#                 matched[distances.index(sorted_distances[0])].append(i)
#         return matched
#
#     def get_pred_html(self, pred_structures, matched_index, ocr_contents):
#         end_html = []
#         td_index = 0
#         for tag in pred_structures:
#             if "</td>" not in tag:
#                 end_html.append(tag)
#                 continue
#
#             if "<td></td>" == tag:
#                 end_html.extend("<td>")
#
#             if td_index in matched_index.keys():
#                 b_with = False
#                 if (
#                         "<b>" in ocr_contents[matched_index[td_index][0]]
#                         and len(matched_index[td_index]) > 1
#                 ):
#                     b_with = True
#                     end_html.extend("<b>")
#
#                 for i, td_index_index in enumerate(matched_index[td_index]):
#                     content = ocr_contents[td_index_index][0]
#                     if len(matched_index[td_index]) > 1:
#                         if len(content) == 0:
#                             continue
#
#                         if content[0] == " ":
#                             content = content[1:]
#
#                         if "<b>" in content:
#                             content = content[3:]
#
#                         if "</b>" in content:
#                             content = content[:-4]
#
#                         if len(content) == 0:
#                             continue
#
#                         if i != len(matched_index[td_index]) - 1 and " " != content[-1]:
#                             content += " "
#                     end_html.extend(content)
#
#                 if b_with:
#                     end_html.extend("</b>")
#
#             if "<td></td>" == tag:
#                 end_html.append("</td>")
#             else:
#                 end_html.append(tag)
#
#             td_index += 1
#
#         # Filter <thead></thead><tbody></tbody> elements
#         filter_elements = ["<thead>", "</thead>", "<tbody>", "</tbody>"]
#         end_html = [v for v in end_html if v not in filter_elements]
#         return "".join(end_html), end_html
#
#     def _filter_ocr_result(self, pred_bboxes, dt_boxes, rec_res):
#         y1 = pred_bboxes[:, 1::2].min()
#         new_dt_boxes = []
#         new_rec_res = []
#
#         for box, rec in zip(dt_boxes, rec_res):
#             if np.max(box[1::2]) < y1:
#                 continue
#             new_dt_boxes.append(box)
#             new_rec_res.append(rec)
#         return new_dt_boxes, new_rec_res
#
# class TablePredictor:
#     def __init__(self, model_dir, model_prefix="inference"):
#         model_file = f"{model_dir}/{model_prefix}.pdmodel"
#         params_file = f"{model_dir}/{model_prefix}.pdiparams"
#         config = Config(model_file, params_file)
#         config.disable_gpu()
#         config.disable_glog_info()
#         config.enable_new_ir(True)
#         config.enable_new_executor(True)
#         config.enable_memory_optim()
#         config.switch_ir_optim(True)
#         # Disable feed, fetch OP, needed by zero_copy_run
#         config.switch_use_feed_fetch_ops(False)
#         predictor = create_predictor(config)
#         self.config = config
#         self.predictor = predictor
#         # Get input and output handlers
#         input_names = predictor.get_input_names()
#         self.input_names = input_names.sort()
#         self.input_handlers = []
#         self.output_handlers = []
#         for input_name in input_names:
#             input_handler = predictor.get_input_handle(input_name)
#             self.input_handlers.append(input_handler)
#         self.output_names = predictor.get_output_names()
#         for output_name in self.output_names:
#             output_handler = predictor.get_output_handle(output_name)
#             self.output_handlers.append(output_handler)
#
#     def __call__(self, batch_imgs):
#         self.input_handlers[0].reshape(batch_imgs.shape)
#         self.input_handlers[0].copy_from_cpu(batch_imgs)
#         self.predictor.run()
#         output = []
#         for out_tensor in self.output_handlers:
#             batch = out_tensor.copy_to_cpu()
#             output.append(batch)
#         return self.format_output(output)
#
#     def format_output(self, pred):
#         return [res for res in zip(*pred)]
#
#
# class SLANetPlus:
#     def __init__(self, model_dir, model_prefix="inference"):
#         self.mean=[0.485, 0.456, 0.406]
#         self.std=[0.229, 0.224, 0.225]
#         self.target_img_size = [488, 488]
#         self.scale=1 / 255
#         self.order="hwc"
#         self.img_loader = LoadImage()
#         self.target_size = 488
#         self.pad_color = 0
#         self.predictor = TablePredictor(model_dir, model_prefix)
#         dict_character=['sos', '<thead>', '</thead>', '<tbody>', '</tbody>', '<tr>', '</tr>', '<td', '>', '</td>', ' colspan="2"', ' colspan="3"', ' colspan="4"', ' colspan="5"', ' colspan="6"', ' colspan="7"', ' colspan="8"', ' colspan="9"', ' colspan="10"', ' colspan="11"', ' colspan="12"', ' colspan="13"', ' colspan="14"', ' colspan="15"', ' colspan="16"', ' colspan="17"', ' colspan="18"', ' colspan="19"', ' colspan="20"', ' rowspan="2"', ' rowspan="3"', ' rowspan="4"', ' rowspan="5"', ' rowspan="6"', ' rowspan="7"', ' rowspan="8"', ' rowspan="9"', ' rowspan="10"', ' rowspan="11"', ' rowspan="12"', ' rowspan="13"', ' rowspan="14"', ' rowspan="15"', ' rowspan="16"', ' rowspan="17"', ' rowspan="18"', ' rowspan="19"', ' rowspan="20"', '<td></td>', 'eos']
#         self.beg_str = "sos"
#         self.end_str = "eos"
#         self.dict = {}
#         self.table_matcher = TableMatch()
#         for i, char in enumerate(dict_character):
#             self.dict[char] = i
#         self.character = dict_character
#         self.td_token = ["<td>", "<td", "<td></td>"]
#
#     def __call__(self, img, ocr_result):
#         img = self.img_loader(img)
#         h, w = img.shape[:2]
#         n_img, h_resize, w_resize = self.resize(img)
#         n_img = self.normalize(n_img)
#         n_img = self.pad(n_img)
#         n_img = n_img.transpose((2, 0, 1))
#         n_img = np.expand_dims(n_img, axis=0)
#         start = time.time()
#         batch_output = self.predictor(n_img)
#         elapse_time = time.time() - start
#         ori_img_size = [[w, h]]
#         output = self.decode(batch_output, ori_img_size)[0]
#         corners = np.stack(output['bbox'], axis=0)
#         dt_boxes, rec_res = get_boxes_recs(ocr_result, h, w)
#         pred_html = self.table_matcher(output['structure'], convert_corners_to_bounding_boxes(corners), dt_boxes, rec_res)
#         return pred_html,output['bbox'], elapse_time
#     def resize(self, img):
#         h, w = img.shape[:2]
#         scale = self.target_size / max(h, w)
#         h_resize = round(h * scale)
#         w_resize = round(w * scale)
#         resized_img = cv2.resize(img, (w_resize, h_resize), interpolation=cv2.INTER_LINEAR)
#         return resized_img, h_resize, w_resize
#     def pad(self, img):
#         h, w = img.shape[:2]
#         tw, th = self.target_img_size
#         ph = th - h
#         pw = tw - w
#         pad = (0, ph, 0, pw)
#         chns = 1 if img.ndim == 2 else img.shape[2]
#         im = cv2.copyMakeBorder(img, *pad, cv2.BORDER_CONSTANT, value=(self.pad_color,) * chns)
#         return im
#     def normalize(self, img):
#         img = img.astype("float32", copy=False)
#         img *= self.scale
#         img -= self.mean
#         img /= self.std
#         return img
#
#
#     def decode(self, pred, ori_img_size):
#         bbox_preds, structure_probs = [], []
#         for bbox_pred, stru_prob in pred:
#             bbox_preds.append(bbox_pred)
#             structure_probs.append(stru_prob)
#         bbox_preds = np.array(bbox_preds)
#         structure_probs = np.array(structure_probs)
#
#         bbox_list, structure_str_list, structure_score = self.decode_single(
#             structure_probs, bbox_preds, [self.target_img_size], ori_img_size
#         )
#         structure_str_list = [
#             (
#                     ["<html>", "<body>", "<table>"]
#                     + structure
#                     + ["</table>", "</body>", "</html>"]
#             )
#             for structure in structure_str_list
#         ]
#         return [
#             {"bbox": bbox, "structure": structure, "structure_score": structure_score}
#             for bbox, structure in zip(bbox_list, structure_str_list)
#         ]
#
#
#     def decode_single(self, structure_probs, bbox_preds, padding_size, ori_img_size):
#         """convert text-label into text-index."""
#         ignored_tokens = [self.beg_str, self.end_str]
#         end_idx = self.dict[self.end_str]
#
#         structure_idx = structure_probs.argmax(axis=2)
#         structure_probs = structure_probs.max(axis=2)
#
#         structure_batch_list = []
#         bbox_batch_list = []
#         batch_size = len(structure_idx)
#         for batch_idx in range(batch_size):
#             structure_list = []
#             bbox_list = []
#             score_list = []
#             for idx in range(len(structure_idx[batch_idx])):
#                 char_idx = int(structure_idx[batch_idx][idx])
#                 if idx > 0 and char_idx == end_idx:
#                     break
#                 if char_idx in ignored_tokens:
#                     continue
#                 text = self.character[char_idx]
#                 if text in self.td_token:
#                     bbox = bbox_preds[batch_idx, idx]
#                     bbox = self._bbox_decode(
#                         bbox, padding_size[batch_idx], ori_img_size[batch_idx]
#                     )
#                     bbox_list.append(bbox.astype(int))
#                 structure_list.append(text)
#                 score_list.append(structure_probs[batch_idx, idx])
#             structure_batch_list.append(structure_list)
#             structure_score = np.mean(score_list)
#             bbox_batch_list.append(bbox_list)
#
#         return bbox_batch_list, structure_batch_list, structure_score
#
#     def _bbox_decode(self, bbox, padding_shape, ori_shape):
#
#         pad_w, pad_h = padding_shape
#         w, h = ori_shape
#         ratio_w = pad_w / w
#         ratio_h = pad_h / h
#         ratio = min(ratio_w, ratio_h)
#
#         bbox[0::2] *= pad_w
#         bbox[1::2] *= pad_h
#         bbox[0::2] /= ratio
#         bbox[1::2] /= ratio
#
#         return bbox
#
#
#
