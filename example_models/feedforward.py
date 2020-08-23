import torch
import torch.nn as nn

import vz_pytorch as vzp


class SimpleFeedforward(nn.Module):
    def __init__(self):
        super(SimpleFeedforward, self).__init__()
        self.l1 = nn.Linear(64, 128)
        self.l2 = nn.Linear(128, 128)
        self.classifier = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        features = torch.sigmoid(self.l2(torch.relu(self.l1(x))))
        y = self.softmax(self.classifier(features))
        # with vzp.pause():
        #     vzp.label(y, f"Outputs\nMax: {y.max():.2f}, {y.argmax()}\nMin: {y.min():.2f}, {y.argmin()}")
        return y