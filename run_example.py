import random
import argparse

import torch
import torchvision
import vz_pytorch as vzp

from matplotlib import pyplot as plt

from vzlogger import connect, get_logger
from torch.utils.tensorboard import SummaryWriter

import vizstack as vz
from example_models import SimpleFeedforward, LSTM
from example_models.utils import plt_to_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["feedforward", "lstm", "resnet", "transformer"], required=True, help="Model to log."
    )
    parser.add_argument("--preprocessing", action="store_true", help="Fix mean and variance of inputs.")
    parser.add_argument(
        "--disable-vz-pytorch", action="store_true", help="Execute the graph without tracking in VZ-Pytorch."
    )
    parser.add_argument("--log-additional-items", action="store_true", help="Send additional content to the logger.")
    parser.add_argument("--tensorboard-dir", default=None, help="Log graph to tensorboard in this directory.")

    args = parser.parse_args()

    model, (x_shape, y_shape) = {
        "feedforward": (SimpleFeedforward(), ((1, 64), None)),
        "lstm": (LSTM(), ((1, 3, 64), None)),
        "resnet": (torchvision.models.resnet18().eval(), ((1, 3, 16, 16), None)),
        "transformer": (torch.nn.Transformer(d_model=64, nhead=2).eval(), ((1, 3, 64), (1, 3, 64))),
    }[args.model]
    if not args.disable_vz_pytorch:
        vzp.start(model)
    x = torch.rand(*x_shape)
    if args.preprocessing:
        mean = torch.mean(x)
        vzp.name(mean, "input_mean")
        x_biased = x - mean
        vzp.name(x_biased, "input_shifted")
        sigma = torch.mean(torch.pow(x_biased, 2))
        vzp.name(sigma, "input_variance")
        x = x_biased / torch.sqrt(sigma)
        vzp.name(x, "input_normalized")
    if y_shape is None:
        model_input = (x,)
    else:
        y = torch.rand(*y_shape)
        model_input = (x, y)
    model(*model_input)
    if not args.disable_vz_pytorch:
        graph = vzp.stop()

    if args.tensorboard_dir is not None:
        writer = SummaryWriter(args.tensorboard_dir)
        writer.add_graph(model, model_input)
        writer.close()
    with connect("http://localhost:4000"):
        if not args.disable_vz_pytorch:
            get_logger("main").info(graph)
        if args.log_additional_items:
            plt.plot(range(100), [1 / (x + 1) * (0.5 + random.random() / 2) for x in range(100)])
            plt.title("Training loss")
            plt.ylabel("Loss")
            plt.xlabel("Step")
            plt.tight_layout()
            loss_graph = plt_to_bytes()
            plt.clf()
            get_logger("main").info(vz.Image(loss_graph))
            get_logger("main").info("Training complete!")
            get_logger("main").info(
                vz.KeyValue({"time": vz.Token("172"), "train_acc": vz.Token("62%"), "val_acc": vz.Token("48%")})
            )


if __name__ == "__main__":
    main()
