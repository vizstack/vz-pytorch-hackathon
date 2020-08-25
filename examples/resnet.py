import torch
import torchvision
import vzpytorch
import vzlogger


# Setup logger for this file.
vzlogger.connect("http://localhost:4000")
logger = vzlogger.get_logger("resnet")

# Instantiate model and inputs.
model = torchvision.models.resnet18().eval()
x = torch.rand(1, 3, 16, 16)

# Run model and trace computation graph.
vzpytorch.start(model)
y = model(x)
graph = vzpytorch.stop(model)

# Display graph using logger.
logger.info(graph)