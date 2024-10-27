import os
import sys
import fitz
import numpy as np
import cv2
from typing import Tuple

class InvoiceProcessor:
    def __init__(self, input_dir: str, output_dir: str, scale_factor: int = 10):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.scale_factor = scale_factor
        os.makedirs(self.output_dir, exist_ok=True)

    def pix2np(self, pix) -> np.ndarray:
        nparr = np.frombuffer(pix, dtype=np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def scales_image_to_correct_pixel_dim(self, im: np.ndarray) -> np.ndarray:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _, im = cv2.threshold(im, 210, 230, cv2.THRESH_BINARY)
        height, width = im.shape[:2]
        scale_h = int(height + (height / float(self.scale_factor)) * 2)
        scale_w = int(width + (width / float(self.scale_factor)) * 2)
        return cv2.resize(im, (scale_w, scale_h), interpolation=cv2.INTER_AREA)

    def noise_removal(self, im: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        im = cv2.dilate(im, kernel, iterations=1)
        im = cv2.erode(im, kernel, iterations=1)
        im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
        return cv2.medianBlur(im, 3)

    def removal_H_V_lines(self, im: np.ndarray) -> np.ndarray:
        gray1 = cv2.bitwise_not(im)
        bw = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        horizontal = np.copy(bw)
        vertical = np.copy(bw)
        cols, rows = horizontal.shape[1], vertical.shape[0]
        horizontal_size, vertical_size = cols // 80, rows // 53

        horizontal = cv2.dilate(cv2.erode(horizontal, cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 3))), 
                                cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 3)))
        horizontal = 255 - cv2.morphologyEx(255 - horizontal, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8), iterations=2)

        vertical = cv2.dilate(cv2.erode(vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (3, vertical_size))),
                              cv2.getStructuringElement(cv2.MORPH_RECT, (3, vertical_size)))
        vertical = 255 - cv2.morphologyEx(255 - vertical, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8), iterations=2)

        kernel = np.ones((2, 2), np.uint8)
        return cv2.dilate(im + horizontal + vertical, kernel)

    def thin_font(self, im: np.ndarray) -> np.ndarray:
        kernel = np.ones((2, 2), np.uint8)
        return cv2.erode(cv2.bitwise_not(im), kernel, iterations=1)

    def think_font(self, im: np.ndarray) -> np.ndarray:
        kernel = np.ones((2, 2), np.uint8)
        return cv2.bitwise_not(cv2.dilate(im, kernel, iterations=2))

    def get_skew_angle(self, im: np.ndarray) -> float:
        blur = cv2.GaussianBlur(im, (9, 9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        dilate = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5)), iterations=2)
        contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        angles = [cv2.minAreaRect(cnt)[-1] for cnt in contours]
        median_angle = np.median(sorted(angles))
        return median_angle - median_angle if int(median_angle) <= 360 else median_angle

    def rotate_image(self, im: np.ndarray, angle: float) -> np.ndarray:
        (h, w) = im.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def border(self, im: np.ndarray) -> np.ndarray:
        color = [0, 0, 0]
        return cv2.copyMakeBorder(im, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=color)

    def pdf_to_img(self, pages: Tuple = None):
        for file in os.listdir(self.input_dir):
            filename = os.fsdecode(file)
            if filename.lower().endswith(".pdf"):
                inputpath = os.path.join(self.input_dir, filename)
                pdf_in = fitz.open(inputpath)
                for pg in range(pdf_in.page_count):
                    if pages and str(pg) not in str(pages):
                        continue
                    page = pdf_in[pg]
                    pix = page.get_pixmap(matrix=fitz.Matrix(600 / 96, 600 / 96), alpha=False)
                    img_data = pix.tobytes("png")
                    im = self.pix2np(img_data)
                    im = self.scales_image_to_correct_pixel_dim(im)
                    im = self.noise_removal(im)
                    im = self.removal_H_V_lines(im)
                    im = self.thin_font(im)
                    im = self.think_font(im)
                    angle = self.get_skew_angle(im)
                    im = self.rotate_image(im, -angle)
                    im = self.border(im)
                    im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)[1]
                    output_file = f"{os.path.splitext(filename)[0]}_page_{pg}.png"
                    cv2.imwrite(os.path.join(self.output_dir, output_file), im)
                pdf_in.close()

# Usage
data_dir = r"your directory to your invoices"
output_dir = os.path.join(data_dir, "OUTPUT_DATA")
processor = InvoiceProcessor(data_dir, output_dir)
processor.pdf_to_img()
