import torch
import torch
import timm

model = timm.create_model('mobilevit_s', pretrained=True)
example_input = torch.randn(1, 3, 256, 256)

max_module_expansion_depth = 3

code_contents = """
import torch
from torchvision.models import squeezenet1_1
from torchvista import trace_model

import timm

model = timm.create_model('mobilevit_s', pretrained=True)
example_input = torch.randn(1, 3, 256, 256)

trace_model(model, example_input, max_module_expansion_depth=3)
"""
