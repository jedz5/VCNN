import torch
import time
import numpy as np
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( #input shape (1,28,28)
            nn.Conv2d(in_channels=2, #input height
                      out_channels=16, #n_filter
                     kernel_size=3, #filter size
                     stride=1, #filter step
                     padding=1 #con2d出来的图片大小不变
                     ), #output shape (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) #2x2采样，output shape (16,14,14)

        )
        # self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), #output shape (32,7,7)
        #                           nn.ReLU(),
        #                           nn.MaxPool2d(2))
        # self.out = nn.Linear(32*7*7,10)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = x.view(x.size(0), -1) #flat (batch_size, 32*7*7)
        # output = self.out(x)
        return x

cnn = CNN()
inp = [torch.arange(50,dtype=torch.float) for x in range(10)]
inp = torch.stack(inp).reshape(-1,2,5,5)
out = cnn(inp)



