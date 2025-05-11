import torch
import torch.nn as nn

class DataHolder:
    def __init__(self, x, y):
        self.x = x
        self.y = [(y)]

class IdentityModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 12)

    def forward(self, x):
        a = x['tensor_1']
        b = x['inner']['tensor_2']
        c = (a * b) / a[0]
        d = self.linear(F.relu(a) - b)
        return DataHolder(c, d)
        

model = IdentityModule()
example_input = {
    'tensor_1': torch.randn(2, 5),
    'inner': {
        'tensor_2': torch.randn(2, 5)
    }
}

code_contents = """
class DataHolder:
    def __init__(self, x, y):
        self.x = x
        self.y = [(y)]

class IdentityModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 12)

    def forward(self, x):
        a = x['tensor_1']
        b = x['inner']['tensor_2']
        c = (a * b) / a[0]
        d = self.linear(F.relu(a) - b)
        return DataHolder(c, d)
        

model = IdentityModule()
example_input = {
    'tensor_1': torch.randn(2, 5),
    'inner': {
        'tensor_2': torch.randn(2, 5)
    }
}

trace_model(model, example_input)
"""
