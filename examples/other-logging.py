import random
import matplotlib.pyplot as plt
import vizstack as vz
import vzpytorch
import vzlogger
from examples.utils import plt_to_base64

# Connect logger for this file.
vzlogger.connect("http://localhost:4000")
logger = vzlogger.get_logger("logging")

# Use matplotlib to make a loss graph.
plt.plot(range(100), [1 / (x + 1) * (0.5 + random.random() / 2) for x in range(100)])
plt.title("Training loss")
plt.ylabel("Loss")
plt.xlabel("Step")
plt.tight_layout()
plot = plt_to_base64()
plt.clf()

# Display plot and info using logger.
logger.info(f"Training complete after {100} steps")
logger.info(vz.KeyValue({
    "steps": 100,
    "train_acc": "62%",
    "val_acc": "48%",
}))
logger.info(vz.Image(plot))

# Disconnect logger.
vzlogger.disconnect()