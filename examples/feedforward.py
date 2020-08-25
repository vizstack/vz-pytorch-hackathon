import torch
import torch.nn as nn
import vzpytorch
import vzlogger


class Feedforward(nn.Module):
    def __init__(self):
        super(Feedforward, self).__init__()
        self.l1 = nn.Linear(64, 128)
        self.l2 = nn.Linear(128, 128)
        self.classifier = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        features = torch.sigmoid(self.l2(torch.relu(self.l1(x))))
        y = self.softmax(self.classifier(features))
        return y

# Connect logger for this file.
logger = vzlogger.get_logger("feedforward")
vzlogger.connect("http://localhost:4000")

# Instantiate model and inputs.
model = Feedforward()
x = torch.rand(1, 64)

# Run model and trace computation graph.
vzpytorch.start(model)
y = model(x)
graph = vzpytorch.stop()

# Display graph using logger.
logger.info(graph)

# Disconnect logger.
vzlogger.disconnect()