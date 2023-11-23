# LP_detection
本项目是基于深度学习的车牌识别，其中，车辆检测网络直接使用YOLO侦测。而后，才是使用网络侦测车牌与识别车牌号。车牌的侦测网络，采用的是resnet网络，网络输出检测边框的仿射变换矩阵，可检测任意形状的四边形。车牌号序列模型，采用transformer模型，输出车牌号序列。训练数据集使用公开数据集，测试集使用实地拍摄的照片
