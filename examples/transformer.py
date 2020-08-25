import torch
import torch.nn as nn
import vzpytorch
import vzlogger


# Setup logger for this file.
vzlogger.connect("http://localhost:4000")
logger = vzlogger.get_logger("transformer")

# Instantiate model and inputs.
model = nn.Transformer(d_model=64, nhead=2).eval()
x1 = torch.rand(1, 3, 64)
x2 = torch.rand(1, 3, 64)

# Run model and trace computation graph.
vzpytorch.start(model)
y = model(x1, x2)
graph = vzpytorch.stop(model)

# Display graph using logger.
logger.info(graph)