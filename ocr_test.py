from fake_chs_lp.random_plate import Draw
# from models.ocr_net2 import OcrNet
from ocr_explorer import Explorer
import cv2
import torch

draw = Draw()
explorer = Explorer()
yes = 0
count = 0
for i in range(1000):
    plate, label = draw()
    plate = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)
    plate = cv2.resize(plate, (144, 48))
    cv2.imshow('a', plate)
    a = explorer(plate)
    if a == label:
        yes += 1
    count += 1
    print(a)
    # print(a)
    # cv2.waitKey(0)
print(yes / count, yes, count)
# cv2.waitKey()
