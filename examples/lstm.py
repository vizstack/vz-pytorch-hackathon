import torch
import torch.nn as nn
import vzlogger
import vzpytorch as vzp
import matplotlib.pyplot as plt
from examples.utils import plt_to_base64


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.cell = nn.LSTMCell(64, 128)

    def forward(self, x):
        outputs = []
        h, c = torch.zeros(1, 128), torch.zeros(1, 128)
        for i in range(x.shape[1]):
            vzp.tick()
            inputs = x[:, i, :]
            vzp.name(inputs, f"token{i}")
            h, c = self.cell(inputs, (h, c))
            vzp.name(h, f"hidden{i}")
            vzp.name(c, f"state{i}")
            with vzp.pause():
                plt.hist(h.detach().numpy().flatten())
                vzp.tag_image(h, plt_to_base64())
                plt.clf()
            outputs.append(h)
        y = torch.stack(outputs, dim=1)
        vzp.name(y, f"outputs")
        return y


# Connect logger for this file.
vzlogger.connect("http://localhost:4000")
logger = vzlogger.get_logger("lstm")

# Instantiate model and inputs.
model = LSTM()
x = torch.rand(1, 3, 64)

# Run model and trace computation graph.
vzp.start(model)
y = model(x)
graph = vzp.stop()

# Display graph using logger.
logger.info(graph)

# Disconnect logger.
vzlogger.disconnect()