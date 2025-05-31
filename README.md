# torchvista

An interactive tool to visualize the forward pass of a PyTorch model directly in the notebook—with a single line of code. Works with web-based notebooks like Jupyter, Google Colab and Kaggle.

## ✨ Features

#### Interactive graph with drag and zoom support

![](docs/assets/interactive-graph.gif)

#### Collapsible nodes for hierarchical modules 

![](docs/assets/collapsible-graph.gif)

#### Error-tolerant partial visualization when errors arise
(e.g., shape mismatches) for ease of debugging

![](docs/assets/error-graph.png)

#### Click on nodes to view parameter and attribute info

![](docs/assets/info-popup.png)


## Demos

- Quick Google Colab tutorial 👉 [here](https://colab.research.google.com/drive/1wrWKhpvGiqHhE0Lb1HnFGeOcS4uBqGXw?usp=sharing#scrollTo=tUKHO2YFKi55) (must be logged in to Colab)
- Check out demos 👉 [here](https://sachinhosmani.github.io/torchvista/)

## ⚙️ Usage

Install via pip
```
pip install torchvista
```

Run from your **web-based notebook** (Jupyter, Colab, etc)

```
import torch
import torch.nn as nn

# Import torchvista
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
## API Reference: `trace_model`

    trace_model(model, inputs, max_module_expansion_depth=3, show_non_gradient_nodes=False)

### Parameters

- **model** (*torch.nn.Module*):
  The model instance to trace.

- **inputs** (*Any*):
  Input(s) to be passed to the model. Can be a single input or a tuple of inputs.

- **max_module_expansion_depth** (*int*, optional):
  Maximum depth for expanding nested modules. Default is `3`.

- **show_non_gradient_nodes** (*bool*, optional):
  Whether to show nodes for scalars, tensors, and NumPy arrays that are not part of the gradient graph  
  (typically constants passed into operations or modules). Default is `False`.
