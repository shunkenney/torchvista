# torchvista

An interactive tool to visualize the forward pass of a PyTorch model directly in the notebook—with a single line of code. Works with web-based notebooks like Jupyter and Google Colab.

## ✨ Features

#### Interactive graph with drag and zoom support

![](docs/assets/interactive-graph.gif)

#### Collapsible nodes for hierarchical modules 

![](docs/assets/collapsible-graph.gif)

#### Error-tolerant partial visualization when errors arise (e.g., shape mismatches) for ease of debugging

![](docs/assets/error-graph.png)


## Demos

Check out demos 👉 [here](https://sachinhosmani.github.io/torchvista/)

## ⚙️ Usage

```
pip install torchvista
```

Run from your **web-based notebook** (Jupyter, Colab, etc)

```
import torch
import torch.nn as nn
from torchvista import trace_model

# Define your module
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

# Instantiate the module and tensor input
model = LinearModel()
example_input = torch.randn(2, 10)

# Trace!
trace_model(model, example_input)
```
