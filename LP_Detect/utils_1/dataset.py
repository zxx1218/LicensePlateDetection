from torch.utils.data import Dataset
from fake_chs_lp.random_plate import Draw
from torch import nn
import os
from torchvision.transforms import transforms
from einops import rearrange
import random
import cv2
from utils import enhance, make_label
import numpy
import torch
import ocr_config
import detect_config
import re


class OcrDataSet(Dataset):

    def __init__(self):
        super(OcrDataSet, self).__init__()
        self.dataset = []
        self.draw = Draw()
        for i in range(100000):
            self.dataset.append(1)
        self.smudge = enhance.Smudge()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        plate, label = self.draw()
        target = []
        for i in label:
            target.append(ocr_config.class_name.index(i))
        plate = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)

        '''数据增强'''
        # plate = self.data_to_enhance(plate)

        cv2.imshow('a', plate)
        cv2.imwrite('1.jpg', plate)
        cv2.waitKey()

        image = torch.from_numpy(plate).permute(2, 0, 1) / 255
        # image = self.transformer(image)
        # print(image.shape)
        target_length = torch.tensor(len(target)).long()
        target = torch.tensor(target).reshape(-1).long()
        _target = torch.full(size=(15,), fill_value=0, dtype=torch.long)
        _target[:len(target)] = target

        return image, _target, target_length

    def data_to_enhance(self, plate):
        '''随机污损'''
        plate = self.smudge(plate)
        '''高斯模糊'''
        plate = enhance.gauss_blur(plate)
        '''高斯噪声'''
        plate = enhance.gauss_noise(plate)
        '''增广数据'''
        plate, pts = enhance.augment_sample(plate)
        '''抠出车牌'''
        plate = enhance.reconstruct_plates(plate, [numpy.array(pts).reshape((2, 4))])[0]
        return plate


class DetectDataset(Dataset):

    def __init__(self):
        super(DetectDataset, self).__init__()
        self.dataset = []
        self.draw = Draw()
        self.smudge = enhance.Smudge()
        root = detect_config.image_root
        root = 'D:/BaiduNetdiskDownload/CCPD2019/CCPD2019/ccpd_fn'
        for image_name in os.listdir('D:/BaiduNetdiskDownload/CCPD2019/CCPD2019/ccpd_fn'):
            box = self.get_box(image_name)
            x3, y3, x4, y4, x1, y1, x2, y2 = box
            box = [x1, y1, x2, y2, x4, y4, x3, y3]
            self.dataset.append((f'{root}/{image_name}', box))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image_path, points = self.dataset[item]
        image = cv2.imread(image_path)

        '''更换假车牌'''
        if random.random() < 0.5:
            plate, _ = self.draw()
            plate = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)
            plate = self.smudge(plate)
            image = enhance.apply_plate(image, points, plate)
        [x1, y1, x2, y2, x4, y4, x3, y3] = points
        points = [x1, x2, x3, x4, y1, y2, y3, y4]
        image, pts = enhance.augment_detect(image, points, 208)

        cv2.imshow('a',image)
        cv2.imwrite('1.jpg',image)
        cv2.waitKey()

        image_tensor = torch.from_numpy(image)/255
        image_tensor = rearrange(image_tensor, 'h w c -> c h w')
        label = make_label.object_label(pts,208,16)
        label = torch.from_numpy(label).float()
        return image_tensor,label

    def up_background(self, image):
        '''高斯模糊'''
        image = enhance.gauss_blur(image)
        '''高斯噪声'''
        image = enhance.gauss_noise(image)
        '''随机剪裁'''
        image = enhance.random_cut(image, (208, 208))
        return image

    def data_to_enhance(self, plate):
        '''随机污损'''
        plate = self.smudge(plate)
        '''高斯模糊'''
        plate = enhance.gauss_blur(plate)
        '''高斯噪声'''
        plate = enhance.gauss_noise(plate)
        '''增广数据'''
        plate, pts = enhance.augment_sample(plate)
        '''抠出车牌'''
        plate = enhance.reconstruct_plates(plate, [numpy.array(pts).reshape((2, 4))])[0]
        return plate

    def get_box(self, name):
        # print(name)
        name = re.split('[.&_-]', name)[7:15]
        # print(name)
        # exit()
        name = [int(i) for i in name]
        return name


if __name__ == '__main__':
    data_set = OcrDataSet()
    for i in range(1000):
        data_set[1]
