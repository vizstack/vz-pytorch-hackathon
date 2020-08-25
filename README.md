# VZ-PyTorch
## Installation

Run the setup script to create a virtual environment and install the `vz-logger` CLI using npm.

```bash
> bash setup.sh
```

Requirements:
  - python3.7: Python version 3.7 or higher.
  - virtualenv: Python virtual environment creator.
  - npm: Node.js package manager to install `@vizstack/vz-logger` (CLI tool).

## Usage

In a terminal, start the logger, then open the logging UI in your browser at `http://localhost:4000`.

```bash
> vz-logger
```

Enter the virtual environemnt and run any of the example scripts:

```bash
> source venv/bin/activate
(venv) > python3 examples/feedforward.py
```

Examples:
  - `feedforward.py`: Feedforward model and basic logging.
  - `transformer.py`: Transformer model and basic logging.
  - `resnet.py`: Built-in model from `torchvision`.
  - `lstm.py`: LSTM model and advanced logging (temporal axes, tensor naming, tagging with plots).
  - `tensorboard.py`: Compare to TensorBoard aesthetics and features.
  - `other-logging.py`: Logging diverse kinds of visualizations (graph, text, plots).
  
## Code
The main project code for this hackathon can be found at:
- vz-pytorch (https://github.com/vizstack/vz-pytorch)

The project also builds upon our other open-source projects:
- vizstack (https://github.com/vizstack/vizstack)
- vz-logger (https://github.com/vizstack/vz-logger)
