import torch
import torch.nn as nn
import math

class ConvolutionalModel(nn.Module):

    def __init__(self, input_shape, conv1_width, conv2_width, in_channels, fc1_width, class_count):
        nn.Module.__init__(self);
        self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=5, stride=1, padding=2, bias=True)

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.droput = nn.Dropout(p=0.5)

        fc1_input_size_w = math.floor((input_shape[0] - 3) / 2 + 1)
        fc1_input_size_w = math.floor((fc1_input_size_w - 3) / 2 + 1)
        fc1_input_size_h = math.floor((input_shape[1] - 3) / 2 + 1)
        fc1_input_size_h = math.floor((fc1_input_size_h - 3) / 2 + 1)
        fc1_input_size = int(fc1_input_size_w*fc1_input_size_h*conv2_width)

        self.fc1 = nn.Linear(fc1_input_size, fc1_width, bias=True)
        self.fc2 = nn.Linear(fc1_width, fc1_width//2, bias=True)
        self.fc_logits = nn.Linear(fc1_width//2, class_count, bias=True)

    def forward(self, x):
        h = self.conv1(x)
        h = torch.relu(h)
        h = self.pool1(h)

        h = self.droput(h)

        h = self.conv2(h)
        h = torch.relu(h)
        h = self.pool2(h)

        h = h.view(h.shape[0], -1)
        h = self.fc1(h)
        h = torch.relu(h)
        h = self.fc2(h)
        h = torch.relu(h)

        logits = self.fc_logits(h)
        return logits