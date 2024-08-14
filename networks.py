import torch
import torch.nn as nn


class GestureCNN(nn.Module):
    def __init__(self, c1, c2, l1, ks):
        super(GestureCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c1, kernel_size=(ks, 3), stride=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=(ks, 3), stride=1, padding='same')
        self.fc1 = nn.Linear(c2 * 21 * 3, l1)
        self.fc2 = nn.Linear(l1, 41)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        #         x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        #         x = self.maxpool(x)

        x = x.view(-1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        return x
