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
            inputs = x[:, i, :]
            vzp.name(inputs, f"token{i}")
            if h is None:
                h, c = self.cell(inputs)
            else:
                h, c = self.cell(inputs, (h, c))
            vzp.name(h, f"hidden{i}")
            vzp.name(c, f"state{i}")
            with vzp.pause():
                plt.hist(h.detach().numpy().flatten())
                vzp.tag_image(h, plt_to_bytes())
                plt.clf()
            outputs.append(h)
        y = torch.stack(outputs, dim=1)
        vzp.name(y, f"output_embeddings")
        return y
