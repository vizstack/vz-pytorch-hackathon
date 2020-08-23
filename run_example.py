import argparse

import torch
import torchvision
import vz_pytorch as vzp

from vzlogger import connect, get_logger

from example_models import SimpleFeedforward, LSTM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["feedforward", "lstm", "resnet", "transformer"], required=True)
    parser.add_argument("--shift-mean", action="store_true")

    args = parser.parse_args()

    model, (x_shape, y_shape) = {
        "feedforward": (SimpleFeedforward(), ((1, 64), None)),
        "lstm": (LSTM(), ((1, 3, 64), None)),
        "resnet": (torchvision.models.resnet18().eval(), ((1, 3, 16, 16), None)),
        "transformer": (torch.nn.Transformer(d_model=64, nhead=2).eval(), ((1, 3, 64), (1, 3, 64)))
    }[args.model]
    vzp.start(model)
    x = torch.rand(*x_shape)
    if y_shape is not None:
        y = torch.rand(*y_shape)
    if args.shift_mean:
        x = x - torch.mean(x)
    if y_shape is None:
        model(x)
    else:
        model(x, y)
    graph = vzp.stop()
    with connect("http://localhost:4000"):
        get_logger("main").info(graph)


if __name__ == "__main__":
    main()
