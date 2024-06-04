from detect_explorer import DExplorer
from ocr_explorer import Explorer
import cv2
import numpy


class ReadPlate:
    """
    读取车牌号
    传入侦测到的车辆图片，即可识别车牌号。
    返回：
        [[车牌号，回归框],……]
    """
    def __init__(self):
        self.detect_exp = DExplorer()
        self.ocr_exp = Explorer()

    def __call__(self, image):
        points = self.detect_exp(image)
        h, w, _ = image.shape
        result = []
        # print(points)
        for point, _ in points:
            plate, box = self.cutout_plate(image, point)
            # print(box)
            lp = self.ocr_exp(plate)
            result.append([lp, box])
            # cv2.imshow('b', plate)
            # cv2.waitKey()
        return result

    def cutout_plate(self, image, point):
        h, w, _ = image.shape
        x1, x2, x3, x4, y1, y2, y3, y4 = point.reshape(-1)
        x1, x2, x3, x4 = x1 * w, x2 * w, x3 * w, x4 * w
        y1, y2, y3, y4 = y1 * h, y2 * h, y3 * h, y4 * h
        src = numpy.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]], dtype="float32")
        dst = numpy.array([[0, 0], [144, 0], [0, 48], [144, 48]], dtype="float32")
        box = [min(x1, x2, x3, x4), min(y1, y2, y3, y4), max(x1, x2, x3, x4), max(y1, y2, y3, y4)]
        M = cv2.getPerspectiveTransform(src, dst)
        out_img = cv2.warpPerspective(image, M, (144, 48))
        return out_img, box


if __name__ == '__main__':
    read_plate = ReadPlate()
    image = cv2.imread('test_image.jpg')
    # image = cv2.imread('2.png')
    boxes = read_plate(image)
    print(boxes)