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
            vzp.name(h, f"hidden{i}")
            vzp.name(c, f"state{i}")
            with vzp.pause():
                vzp.tag(h, f"Mean: {h.mean():.2f}")
                plt.hist(h.detach().numpy().flatten())
                vzp.tag_image(h, plt_to_bytes())
                plt.clf()
            outputs.append(h)
        return torch.stack(outputs, dim=1)