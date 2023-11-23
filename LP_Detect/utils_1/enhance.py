import cv2
import numpy
from random import randint
import random
import math
import sys

sys.path.append('./utils')


class Smudge:
    """
    随机污损
    """

    def __init__(self, smu='D:/deep_project/lp/utils/smu.png'):
        self._smu = cv2.imread(smu)

    def __call__(self, image):
        # print(self._smu.shape)
        h1, w1, _ = self._smu.shape
        h2, w2, _ = image.shape
        y = randint(0, h1 - h2)
        x = randint(0, w1 - w2)
        texture = self._smu[y:y + h2, x:x + w2]
        return cv2.bitwise_not(cv2.bitwise_and(cv2.bitwise_not(image), texture))


def gauss_blur(image):
    level = randint(0, 8)
    return cv2.blur(image, (level * 2 + 1, level * 2 + 1))


def gauss_noise(image):
    for i in range(image.shape[2]):
        c = image[:, :, i]
        diff = 255 - c.max()
        noise = numpy.random.normal(0, randint(1, 6), c.shape)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        noise = diff * noise
        image[:, :, i] = c + noise.astype(numpy.uint8)
    return image


def transform_matrix(pts, t_pts):
    return cv2.getPerspectiveTransform(numpy.float32(pts[:2, :].T), numpy.float32(t_pts[:2, :].T))


def points_matrix(pts):
    return numpy.matrix(numpy.concatenate((pts, numpy.ones((1, pts.shape[1]))), 0))


def rect_matrix(tlx, tly, brx, bry):
    return numpy.matrix([
        [tlx, brx, brx, tlx],
        [tly, tly, bry, bry],
        [1.0, 1.0, 1.0, 1.0]
    ])


def rotate_matrix(width, height, angles=numpy.zeros(3), zcop=1000.0, dpp=1000.0):
    rads = numpy.deg2rad(angles)
    rx = numpy.matrix([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(rads[0]), math.sin(rads[0])],
        [0.0, -math.sin(rads[0]), math.cos(rads[0])]
    ])
    ry = numpy.matrix([
        [math.cos(rads[1]), 0.0, -math.sin(rads[1])],
        [0.0, 1.0, 0.0],
        [math.sin(rads[1]), 0.0, math.cos(rads[1])]
    ])
    rz = numpy.matrix([
        [math.cos(rads[2]), math.sin(rads[2]), 0.0],
        [-math.sin(rads[2]), math.cos(rads[2]), 0.0],
        [0.0, 0.0, 1.0]
    ])
    r = rx * ry * rz
    hxy = numpy.matrix([
        [0.0, 0.0, width, width],
        [0.0, height, 0.0, height],
        [1.0, 1.0, 1.0, 1.0]
    ])
    xyz = numpy.matrix([
        [0.0, 0.0, width, width],
        [0.0, height, 0.0, height],
        [0.0, 0.0, 0.0, 0.0]
    ])
    half = numpy.matrix([[width], [height], [0.0]]) / 2.0
    xyz = r * (xyz - half) - numpy.matrix([[0.0], [0.0], [zcop]])
    xyz = numpy.concatenate((xyz, numpy.ones((1, 4))), 0)
    p = numpy.matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0 / dpp, 0.0]
    ])
    t_hxy = p * xyz
    t_hxy = t_hxy / t_hxy[2, :] + half
    return transform_matrix(hxy, t_hxy)


def project(img, pts, trans, dims):
    t_img = cv2.warpPerspective(img, trans, (dims, dims))
    t_pts = numpy.matmul(trans, points_matrix(pts))
    t_pts = t_pts / t_pts[2]
    return t_img, t_pts[:2]


def hsv_noise(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = hsv[:, :, 0] * (0.8 + random.uniform(0.0, 0.8))
    hsv[:, :, 1] = hsv[:, :, 1] * (0.3 + random.uniform(0.0, 0.7))
    hsv[:, :, 2] = hsv[:, :, 2] * (0.2 + random.uniform(0.0, 0.2))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def brightness_noise(img, ratio=0.1):
    return numpy.clip(img * (1.0 + random.uniform(-ratio, ratio)), 0, 255)


def update(image, lightness, saturation):
    # 颜色空间转换 BGR转为HLS
    image = image.astype(numpy.float32) / 255.0
    hlsImg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # 1.调整亮度（线性变换)
    hlsImg[:, :, 1] = (1.0 + lightness / float(100)) * hlsImg[:, :, 1]
    hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1
    # 饱和度
    hlsImg[:, :, 2] = (1.0 + saturation / float(100)) * hlsImg[:, :, 2]
    hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
    # HLS2BGR
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255
    lsImg = lsImg.astype(numpy.uint8)
    return lsImg


def augment_sample(image, dims=208):
    points = [val + random.uniform(-0.1, 0.1) for val in [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]]
    # image = image.astype("uint8").asnumpy()
    points = numpy.array(points).reshape((2, 4))
    points = points * numpy.array([[image.shape[1]], [image.shape[0]]])
    # random crop
    wh_ratio = random.uniform(2.0, 4.0)
    width = random.uniform(dims * 0.2, dims * 1.0)
    height = width / wh_ratio
    dx = random.uniform(0.0, dims - width)
    dy = random.uniform(0.0, dims - height)
    crop = transform_matrix(
        points_matrix(points),
        rect_matrix(dx, dy, dx + width, dy + height)
    )
    # random rotate
    max_angles = numpy.array([80.0, 80.0, 45.0])
    angles = numpy.random.rand(3) * max_angles
    if angles.sum() > 120:
        angles = (angles / angles.sum()) * (max_angles / max_angles.sum())
    rotate = rotate_matrix(dims, dims, angles)
    # apply projection
    image, points = project(image, points, numpy.matmul(rotate, crop), dims)
    # scale the coordinates of points to [0, 1]
    points = points / dims
    # color augment
    image = hsv_noise(image)
    # brightness augment
    image = update(image, randint(-80, 100), randint(-80, 100))
    # image = brightness_noise(image)
    return image, numpy.asarray(points).reshape((-1,)).tolist()


def reconstruct_plates(image, plate_pts, out_size=(144, 48)):
    wh = numpy.array([[image.shape[1]], [image.shape[0]]])
    plates = []
    for pts in plate_pts:
        pts = points_matrix(pts * wh)
        t_pts = rect_matrix(0, 0, out_size[0], out_size[1])
        m = transform_matrix(pts, t_pts)
        plate = cv2.warpPerspective(image, m, out_size)
        plates.append(plate)
    return plates


def random_cut(image, size):
    h, w, c = image.shape
    min_side = min(h, w)
    h_sid_len = random.randint(int(0.2 * min_side), int(0.9 * min_side))
    w_sid_len = random.randint(int(0.2 * min_side), int(0.9 * min_side))
    h_s = random.randint(0, h - h_sid_len)
    w_s = random.randint(0, w - w_sid_len)
    image = image[h_s:h_s + h_sid_len, w_s:w_s + w_sid_len]
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image


def apply_plate(image, points, plate):
    points = [[points[2 * i], points[2 * i + 1]] for i in range(4)]
    points = numpy.float32(points)
    h, w, _ = plate.shape
    pt2 = numpy.float32([[0, 0], [w, 0], [0, h], [w, h]])
    m = cv2.getPerspectiveTransform(pt2, points)
    h, w, _ = image.shape
    mask = numpy.ones_like(plate, dtype=numpy.uint8)
    out_img = cv2.warpPerspective(plate, m, (w, h))
    mask = cv2.warpPerspective(mask, m, (w, h))
    mask = mask != 0
    image[mask] = out_img[mask]
    # cv2.imshow('c', image)
    return image


def augment_detect(image, points, dims, flip_prob=0.5):
    points = numpy.array(points).reshape((2, 4))
    wh_ratio = random.uniform(2.0, 4.0)
    width = random.uniform(dims * 0.2, dims * 1.0)
    height = width / wh_ratio
    dx = random.uniform(0.0, dims - width)
    dy = random.uniform(0.0, dims - height)
    crop = transform_matrix(
        points_matrix(points),
        rect_matrix(dx, dy, dx + width, dy + height)
    )
    # random rotate
    max_angles = numpy.array([80.0, 80.0, 45.0])
    angles = numpy.random.rand(3) * max_angles
    if angles.sum() > 120:
        angles = (angles / angles.sum()) * (max_angles / max_angles.sum())
    # print(angles)
    rotate = rotate_matrix(dims, dims, angles)
    # apply projection
    image, points = project(image, points, numpy.matmul(rotate, crop), dims)
    # scale the coordinates of points to [0, 1]
    points = points / dims
    # random flip
    if random.random() < flip_prob:
        image = cv2.flip(image, 1)
        points[0] = 1 - points[0]
        points = points[..., [1, 0, 3, 2]]
        # print(11111)
    # color augment
    image = hsv_noise(image)
    # brightness augment
    image = update(image, randint(-80, 300), randint(-80, 250))
    return image, numpy.asarray(points).reshape((-1,)).tolist()
