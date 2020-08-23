# VZ-Pytorch
## Usage
First, install dependencies and launch a local logging server by running
```bash
bash setup.sh
```
Then, open `http://localhost:4000` in your browser to begin viewing logs. You can generate logs by running
```bash
python run_example.py --model lstm
```
The `--model` argument currently supports `feedforward`, `lstm`, `resnet`, and `transformer`.
