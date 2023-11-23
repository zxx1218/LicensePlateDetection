# import argparse
import os
import sys
from pathlib import Path
import numpy
import torch
# import torch.backends.cudnn as cudnn
from read_plate import ReadPlate

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
# from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from PIL import Image, ImageDraw, ImageFont


class Kenshutsu(object):

    def __init__(self, is_cuda):
        device = '0' if is_cuda and torch.cuda.is_available() else 'cpu'
        weights = './weights/yolov5s (1).pt'
        if not os.path.exists(weights):
            raise RuntimeError('Model parameters not found')
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device)
        imgsz = (640, 640)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(imgsz, s=stride)
        bs = 1
        self.model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        self.agnostic_nms = False
        self.classes = None
        self.iou_thres = 0.45
        self.conf_thres = 0.25

    def __call__(self, image):
        h, w, c = image.shape
        image, h2, w2, fx = self.square_picture(image, 640)
        image_tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = numpy.transpose(image_tensor, axes=(2, 0, 1)) / 255
        image_tensor = torch.from_numpy(image_tensor).float().to(self.device)
        image_tensor = image_tensor.unsqueeze(0)
        pred = self.model(image_tensor)
        pred = \
        non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=1000)[0]
        boxes = pred.cpu()
        result = []
        for box in boxes:
            x1, y1, x2, y2, the, c = box
            x1, y1, x2, y2 = max(0, int((x1 - (640 // 2 - w2 // 2)) / fx)), max(0, int((y1 - (
                        640 // 2 - h2 // 2)) / fx)), min(w, int((x2 - (640 // 2 - w2 // 2)) / fx)), min(h, int((y2 - (
                        640 // 2 - h2 // 2)) / fx))
            result.append([x1, y1, x2, y2, the, c])
        return result

    @staticmethod
    def square_picture(image, image_size):
        h1, w1, _ = image.shape
        max_len = max(h1, w1)
        if max_len >= image_size:
            fx = image_size / max_len
            fy = image_size / max_len
            image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
            h2, w2, _ = image.shape
            background = numpy.zeros((image_size, image_size, 3), dtype=numpy.uint8)
            background[:, :, :] = 127
            s_h = image_size // 2 - h2 // 2
            s_w = image_size // 2 - w2 // 2
            background[s_h:s_h + h2, s_w:s_w + w2] = image
            return background, h2, w2, fx
        else:
            h2, w2, _ = image.shape
            background = numpy.zeros((image_size, image_size, 3), dtype=numpy.uint8)
            background[:, :, :] = 127
            s_h = image_size // 2 - h2 // 2
            s_w = image_size // 2 - w2 // 2
            background[s_h:s_h + h2, s_w:s_w + w2] = image
            return background, h2, w2, 1


def DrawChinese(img, text, positive, fontSize=20, fontColor=(
        255, 0, 0)):  # args-(img:numpy.ndarray, text:中文文本, positive:位置, fontSize:字体大小默认20, fontColor:字体颜色默认绿色)
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)
    # PIL图片上打印汉字
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("MSJHL.TTC", fontSize, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
    draw.text(positive, text, fontColor, font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体格式
    print(text)
    cv2charimg = cv2.cvtColor(numpy.array(pilimg), cv2.COLOR_RGB2BGR)  # PIL图片转cv2 图片
    return cv2charimg


if __name__ == '__main__':
    import os

    class_name = ['main']
    root = 'test_image'
    detecter = Kenshutsu(False)
    read_plate = ReadPlate()
    count = 0

    for image_name in os.listdir(root):
        image_path = f'{root}/{image_name}'
        image = cv2.imread(image_path)
        boxes = detecter(image)
        plates = []
        for box in boxes:
            x1, y1, x2, y2, the, c = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2, the, c)
            if c == 2 or c == 5:
                image_ = image[y1:y2, x1:x2]
                result = read_plate(image_)
                if result:
                    plate, (x11, y11, x22, y22) = result[0]
                    plates.append((x1, y1, x2, y2, plate, x11 + x1, y11 + y1, x22 + x1, y22 + y1))
        for plate in plates:
            x1, y1, x2, y2, plate_name, x11, y11, x22, y22 = plate
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x11, y11, x22, y22 = int(x11), int(y11), int(x22), int(y22)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            image = cv2.rectangle(image, (x11 - 5, y11 - 5), (x22 + 5, y22 + 5), (0, 0, 255), 2)
            image = DrawChinese(image, plate_name, (x11, y22), 30)
        #
        # image = cv2.resize(image, None, fx=0.5, fy=0.5)
        print(image_name)
        cv2.imshow('a', image)
        cv2.waitKey()
