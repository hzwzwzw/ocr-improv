from io import BytesIO
from pathlib import Path
from typing import Union, List

import numpy as np
import cv2
from PIL import UnidentifiedImageError, Image

InputType = Union[str, np.ndarray, bytes, Path, Image.Image]


class LoadImage:
    def __init__(
            self,
    ):
        pass

    def __call__(self, img: InputType) -> np.ndarray:
        if not isinstance(img, InputType.__args__):
            raise LoadImageError(
                f"The img type {type(img)} does not in {InputType.__args__}"
            )

        origin_img_type = type(img)
        img = self.load_img(img)
        img = self.convert_img(img, origin_img_type)
        return img

    def load_img(self, img: InputType) -> np.ndarray:
        if isinstance(img, (str, Path)):
            self.verify_exist(img)
            try:
                img = np.array(Image.open(img))
            except UnidentifiedImageError as e:
                raise LoadImageError(f"cannot identify image file {img}") from e
            return img

        if isinstance(img, bytes):
            img = np.array(Image.open(BytesIO(img)))
            return img
        if isinstance(img, BytesIO):
            img = np.array(Image.open(img))
            return img
        if isinstance(img, np.ndarray):
            return img

        if isinstance(img, Image.Image):
            return np.array(img)

        raise LoadImageError(f"{type(img)} is not supported!")

    def convert_img(self, img: np.ndarray, origin_img_type):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 3:
            channel = img.shape[2]
            if channel == 1:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if channel == 2:
                return self.cvt_two_to_three(img)

            if channel == 3:
                if issubclass(origin_img_type, (str, Path, bytes, Image.Image)):
                    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img

            if channel == 4:
                return self.cvt_four_to_three(img)

            raise LoadImageError(
                f"The channel({channel}) of the img is not in [1, 2, 3, 4]"
            )

        raise LoadImageError(f"The ndim({img.ndim}) of the img is not in [2, 3]")

    @staticmethod
    def cvt_two_to_three(img: np.ndarray) -> np.ndarray:
        """gray + alpha → BGR"""
        img_gray = img[..., 0]
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        img_alpha = img[..., 1]
        not_a = cv2.bitwise_not(img_alpha)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(img_bgr, img_bgr, mask=img_alpha)
        new_img = cv2.add(new_img, not_a)
        return new_img

    @staticmethod
    def cvt_four_to_three(img: np.ndarray) -> np.ndarray:
        """RGBA → BGR"""
        r, g, b, a = cv2.split(img)
        new_img = cv2.merge((b, g, r))

        not_a = cv2.bitwise_not(a)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(new_img, new_img, mask=a)
        new_img = cv2.add(new_img, not_a)
        return new_img

    @staticmethod
    def verify_exist(file_path: Union[str, Path]):
        if not Path(file_path).exists():
            raise LoadImageError(f"{file_path} does not exist.")


class LoadImageError(Exception):
    pass


def plot_rec_box_with_logic_info(img_path, logic_points, sorted_polygons, without_text=True):
    """
    :param img_path
    :param output_path
    :param logic_points: [row_start,row_end,col_start,col_end]
    :param sorted_polygons: [xmin,ymin,xmax,ymax]
    :return:
    """
    # 读取原图
    img = cv2.imread(img_path)
    img = cv2.copyMakeBorder(
        img, 0, 0, 0, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    # 绘制 polygons 矩形
    for idx, polygon in enumerate(sorted_polygons):
        x0, y0, x1, y1 = polygon[0], polygon[1], polygon[2], polygon[3]
        x0 = round(x0)
        y0 = round(y0)
        x1 = round(x1)
        y1 = round(y1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
        # 增大字体大小和线宽
        font_scale = 1.0  # 原先是0.5
        thickness = 2  # 原先是1
        if without_text:
            return img
        cv2.putText(
            img,
            f"{idx}",
            (x1, y1),
            cv2.FONT_HERSHEY_PLAIN,
            font_scale,
            (0, 0, 255),
            thickness,
        )
        return img


def plot_rec_box(img, sorted_polygons):
    """
    :param img_path
    :param output_path
    :param sorted_polygons: [xmin,ymin,xmax,ymax]
    :return:
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 处理ocr_res
    img = cv2.copyMakeBorder(
        img, 0, 0, 0, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    # 绘制 ocr_res 矩形
    for idx, polygon in enumerate(sorted_polygons):
        x0, y0, x1, y1 = polygon[0], polygon[1], polygon[2], polygon[3]
        x0 = round(x0)
        y0 = round(y0)
        x1 = round(x1)
        y1 = round(y1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
        # 增大字体大小和线宽
        font_scale = 1.0  # 原先是0.5
        thickness = 2  # 原先是1

        # cv2.putText(
        #     img,
        #     str(idx),
        #     (x1, y1),
        #     cv2.FONT_HERSHEY_PLAIN,
        #     font_scale,
        #     (0, 0, 255),
        #     thickness,
        # )
    return img

def format_html(html:str):
    html = html.replace("<html>","")
    html = html.replace("</html>","")
    html = html.replace("<body>", "")
    html = html.replace("</body>", "")
    return f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
    <meta charset="UTF-8">
    <title>Complex Table Example</title>
    <style>
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        th, td {{
            border: 1px solid black;
            padding: 8px;
            text-align: center;
        }}
        th {{
            background-color: #f2f2f2;
        }}
    </style>
    </head>
    <body>
    {html}
    </body>
    </html>
    """

def box_4_2_poly_to_box_4_1(poly_box: Union[np.ndarray, list]) -> List[float]:
    """
    将poly_box转换为box_4_1
    :param poly_box:
    :return:
    """
    return [poly_box[0][0], poly_box[0][1], poly_box[2][0], poly_box[2][1]]