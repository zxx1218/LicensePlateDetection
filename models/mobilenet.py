from torchvision.models import mobilenet_v3_small
from torch import nn
import torch


class MobileNet(nn.Module):

    def __init__(self, out_features):
        super(MobileNet, self).__init__()
        self.headbone = mobilenet_v3_small(True)
        # print(self.headbone)
        # exit()
        self.dropout = nn.Dropout(0.1)
        self.headbone.classifier[3] =  nn.Linear(1024, out_features)
        # print('---------------------')
        # print(self.headbone)

    def forward(self, x):
        x = self.dropout(x)
        x = self.headbone(x)
        return x


if __name__ == '__main__':
    m = MobileNet(3)
    # print(m)
    x = torch.randn(2, 3, 1080//3, 1920//3)
    print(m(x).shape)
