import torch
import torchvision
import torch.utils.tensorboard as tensorboard


# Instantiate model and inputs.
model = torchvision.models.resnet18().eval()
x = torch.rand(1, 3, 16, 16)

# Run model.
y = model(x)

# Display graph using TensorBoard.
writer = tensorboard.SummaryWriter()
writer.add_graph(model, x)
writer.close()