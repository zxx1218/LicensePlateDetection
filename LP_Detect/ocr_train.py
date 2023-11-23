import ocr_config
import torch
from torch import nn
from torch.utils.data import DataLoader
from models_1.ocr_net2 import OcrNet
from utils_1.dataset import OcrDataSet
import os
from tqdm import tqdm


class Trainer:

    def __init__(self, load_parameters=False):
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print('准备使用设备%s训练网络' % self.device)
        self.net = OcrNet(ocr_config.num_class)
        if os.path.exists(ocr_config.weight) and load_parameters:
            self.net.load_state_dict(torch.load(ocr_config.weight, map_location='cpu'))
            print('成功加载模型参数')
        # elif not load_parameters:
        #     print('未能加载模型参数')
        # else:
        #     raise RuntimeError('Model parameters are not loaded')
        self.dataset = OcrDataSet()
        self.dataloader = DataLoader(self.dataset, 512, True)
        self.net = self.net.to(self.device).train()
        self.loss_func = nn.CTCLoss(blank=0, zero_infinity=True)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.000001)

    def __call__(self):
        # accumulation_steps = 3
        epoch = 0
        print(len(self.dataloader))
        while True:
            loss_sum=0
            for i, (images, targets, target_lengths) in tqdm(enumerate(self.dataloader)):
                images = images.to(self.device)

                '''生成标签'''
                e = torch.tensor([])
                for i, j in enumerate(target_lengths):
                    e = torch.cat((e, targets[i][:j]), dim=0)
                targets = e.long()
                # print(targets)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)

                '''预测'''
                predict = self.net(images)
                s, n, v = predict.shape
                input_lengths = torch.full(size=(n,), fill_value=s, dtype=torch.long)

                """计算损失，预测值需，log_softmax处理，网络部分不应该softmax"""
                loss = self.loss_func(predict.log_softmax(2), targets, input_lengths, target_lengths)

                '''反向传播，梯度更新'''
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                '''统计日志信息'''
                loss_sum += loss.item()
                i+=1
            logs = f'''{epoch},loss_sum: {loss_sum / len(self.dataloader)}'''
            torch.save(self.net.state_dict(), ocr_config.weight)
            print(logs)
            epoch += 1


if __name__ == '__main__':
    trainer = Trainer(True)
    trainer()
