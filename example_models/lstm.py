import torch
import torch.nn as nn

from matplotlib import pyplot as plt

import vz_pytorch as vzp

from .utils import plt_to_bytes


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.cell = nn.LSTMCell(64, 128)

    def forward(self, x):
        outputs = []
        h, c = None, None
        for i in range(x.shape[1]):
            vzp.tick()
            if h is None:
                h, c = self.cell(x[:, i, :])
            else:
                h, c = self.cell(x[:, i, :], (h, c))
            with vzp.pause():
                plt.hist(h.detach().numpy().flatten())
                plt.title("Histogram")
                vzp.tag(h, plt_to_bytes(), "image")
                plt.clf()
                vzp.tag(h, f"Mean: {h.mean():.2f}", "text")
                vzp.tag(h, f"Max: {h.max():.2f} (index {h.argmax()})", "text")
                vzp.tag(h, f"Min: {h.min():.2f} (index {h.argmin()})", "text")
            outputs.append(h)
        return torch.stack(outputs, dim=1)