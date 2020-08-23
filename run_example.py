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

    model, x_shape = {
        "feedforward": (SimpleFeedforward(), (1, 64)),
        "lstm": (LSTM(), (1, 3, 64)),
        "resnet": (torchvision.models.resnet18().eval(), (1, 3, 16, 16)),
        "transformer": (torch.nn.Transformer(d_model=64, nhead=2).eval(), (1, 3, 64))
    }[args.model]
    vzp.start(model)
    x = torch.rand(*x_shape)
    if args.shift_mean:
        x = x - torch.mean(x)
    model(x)
    graph = vzp.finish()
    with connect("http://localhost:4000"):
        get_logger("main").info(graph)


if __name__ == "__main__":
    main()
