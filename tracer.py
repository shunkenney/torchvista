import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import inspect
from IPython.core.ultratb import AutoFormattedTB

import json
import json
from pathlib import Path
import uuid
from collections import defaultdict

class SimpleTorchTracer:
    STANDARD_TORCH_MODULES = set([
        nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.LSTM, nn.GRU, nn.RNN,
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm,
        nn.Dropout, nn.Dropout2d, nn.Dropout3d,
        nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ELU, nn.SELU, nn.GELU, nn.Tanh,
        nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
        nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
        nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
        nn.Embedding, nn.Flatten
    ])
    TENSOR_FUNCTION_OPS = [
        torch.stack,
        torch.cat,
    ]
    TENSOR_METHOD_OPS = [
        torch.Tensor.__add__,
        torch.Tensor.__iadd__,
        torch.Tensor.__mul__,
        torch.Tensor.__getitem__,
        torch.Tensor.view,
        torch.Tensor.reshape,
        torch.Tensor.sum,
        torch.Tensor.mean,
        torch.Tensor.max,
        torch.Tensor.min,
    ]
    ACTIVATION_FUNCTIONS = [
            F.relu,
            F.leaky_relu,
            F.elu,
            F.selu,
            F.gelu,
            F.sigmoid,
            F.tanh,
            F.softmax,
            F.log_softmax,
            F.max_pool1d,
            F.max_pool2d,
            F.max_pool3d,
            F.avg_pool1d,
            F.avg_pool2d,
            F.avg_pool3d,
            F.dropout,
        ]
    def __init__(self):
        self.adj_list = OrderedDict()
        self.op_type_counters = defaultdict(int)
        self.last_successful_op = None
        self.module_to_node_name = {}
        self.current_op = None
        self.original_ops = {}
        self.module_reuse_count = {}
        self.module_info = {}
        self.func_info_map = {}
        self.node_to_base_name_map = {}
        self.module_hierarchy = {}
        self.traced_modules = set()
        self.current_executing_module = None
        self.module_stack = []
        self.parent_module_to_nodes = defaultdict(list)
        self.parent_module_to_depth = {}

    def format_dims(self, dims):
        if isinstance(dims, tuple):
            return f"({', '.join(map(str, dims))})"
        elif isinstance(dims, list):
            return f"[{', '.join(self.format_dims(d) for d in dims)}]"
        else:
            return str(dims)

    def get_unique_op_name(self, op_type, module=None):
        if module:
            if module not in self.module_to_node_name:
                self.op_type_counters[op_type] += 1
                base_name = f"{op_type}_{self.op_type_counters[op_type]}"
                self.module_to_node_name[module] = base_name
                self.module_info[base_name] = self.get_module_info(module)
                self.node_to_base_name_map[base_name] = base_name
                return base_name, True
            else:
                base_name = self.module_to_node_name[module]
                self.module_reuse_count[base_name] = self.module_reuse_count.get(base_name, 0) + 1
                reused_name = f"{base_name}_Reused_{self.module_reuse_count[base_name]}"
                self.node_to_base_name_map[reused_name] = base_name
                return reused_name, True
        else:
            self.op_type_counters[op_type] += 1
            op_name = f"{op_type}_{self.op_type_counters[op_type]}"
            self.node_to_base_name_map[op_name] = op_name
            return op_name, False

    def get_module_info(self, module):
        info = {
            'type': type(module).__name__,
            'parameters': {},
            'attributes': {},
        }

        for attr_name in dir(module):
            if attr_name.startswith('_') or callable(getattr(module, attr_name)):
                continue
            attr_value = getattr(module, attr_name)
            if isinstance(attr_value, (int, float, str, bool, tuple)):
                info['attributes'][attr_name] = attr_value

        for name, param in module.named_parameters(recurse=False):
            info['parameters'][name] = {
                'shape': tuple(param.shape),
                'requires_grad': param.requires_grad
            }

        if hasattr(module, 'extra_repr') and callable(module.extra_repr):
            info['extra_repr'] = module.extra_repr()

        return info

    def format_arg(self, arg, max_length=50):
        if isinstance(arg, torch.Tensor):
            return f"tensor({','.join(str(d) for d in arg.shape)})"
        elif isinstance(arg, (list, tuple)) and all(isinstance(item, torch.Tensor) for item in arg):
            return str([self.format_arg(item) for item in arg])
        elif isinstance(arg, (int, float, bool)):
            return arg
        else:
            arg_str = str(arg)
            if len(arg_str) > max_length:
                return arg_str[:max_length - 3] + "..."
            return arg_str

    def capture_args(self, *args, **kwargs):
        formatted_args = [self.format_arg(arg) for arg in args]
        formatted_kwargs = {k: self.format_arg(v) for k, v in kwargs.items()}
        return formatted_args, formatted_kwargs

    def pre_trace_op(self, op_type, inputs, module=None, *args, **kwargs):
        op_name, is_module = self.get_unique_op_name(op_type, module)
        input_dims = tuple(inputs[0].shape) if isinstance(inputs[0], torch.Tensor) else \
                     [tuple(t.shape) for t in inputs[0]] if isinstance(inputs[0], (list, tuple)) and all(isinstance(t, torch.Tensor) for t in inputs[0]) \
                     else inputs[0]

        self.adj_list[op_name] = {
            'edges': [],
            'input_dims': self.format_dims(input_dims),
            'output_dims': None,
            'failed': True,
            'is_module': is_module,
        }

        for inp in inputs:
            if isinstance(inp, torch.Tensor) and hasattr(inp, '_tl_name'):
                if inp._tl_name not in self.adj_list[op_name]['edges']:
                    self.adj_list[inp._tl_name]['edges'].append(op_name)
            elif isinstance(inp, (list, tuple)):
                for t in inp:
                    if isinstance(t, torch.Tensor) and hasattr(t, '_tl_name'):
                        if t._tl_name not in self.adj_list[op_name]['edges']:
                            self.adj_list[t._tl_name]['edges'].append(op_name)

        formatted_args, formatted_kwargs = self.capture_args(*args, **kwargs)
        self.func_info_map[op_name] = {}
        self.func_info_map[op_name]["positional_args"] = formatted_args
        self.func_info_map[op_name]["keyword_args"] = formatted_kwargs

        self.current_op = op_name

        depth = 1
        for parent in self.module_stack[::-1]:
            self.parent_module_to_nodes[parent].append(op_name)
            self.parent_module_to_depth[parent] = max(depth, 0 if parent not in self.parent_module_to_depth else self.parent_module_to_depth[parent])
            depth += 1
        
        return op_name

    def trace_op(self, op_name, output):
        output_dims = self.format_dims(tuple(output.shape) if isinstance(output, torch.Tensor) else output)
        self.adj_list[op_name]['output_dims'] = output_dims
        self.adj_list[op_name]['failed'] = False

        if isinstance(output, torch.Tensor):
            output._tl_name = op_name

        self.last_successful_op = op_name
        self.current_op = None
        return output

    def wrap_traced_module(self, module):
        if module in self.traced_modules:
            return
        orig_forward = module.forward

        def wrapped_forward(*args, **kwargs):
            self.current_executing_module = module
            op_name = self.pre_trace_op(type(module).__name__, args, module, *args, **kwargs)
            output = orig_forward(*args, **kwargs)
            result = self.trace_op(op_name, output)
            self.current_executing_module = None
            return result

        module.forward = wrapped_forward
        self.traced_modules.add(module)

    def wrap_untraced_module(self, module):
        if module in self.traced_modules:
            return
        orig_forward = module.forward

        def wrapped_forward(*args, **kwargs):
            self.module_stack.append(self.get_unique_op_name(type(module).__name__, module)[0])
            output = orig_forward(*args, **kwargs)
            self.module_stack.pop()
            return output

        module.forward = wrapped_forward
        self.traced_modules.add(module)

    def traverse_model(self, model, parent=None):
        for name, module in model.named_children():
            self.module_hierarchy[module] = parent
            if type(module) in self.STANDARD_TORCH_MODULES:
                self.wrap_traced_module(module)
            else:
                self.wrap_untraced_module(module)
            if list(model.named_children()):
                self.traverse_model(module, parent=module)

    def wrap_tensor_ops(self):
        def make_wrapped(orig_op, op_name):
            def wrapped(*args, **kwargs):
                if SimpleTorchTracer.instance.current_executing_module is None:
                    node_name = SimpleTorchTracer.instance.pre_trace_op(op_name, args, None, *args, **kwargs)
                    output = orig_op(*args, **kwargs)
                    return SimpleTorchTracer.instance.trace_op(node_name, output)
                else:
                    return orig_op(*args, **kwargs)
            return wrapped

        for op in self.TENSOR_FUNCTION_OPS:
            if op.__name__ not in self.original_ops:
                self.original_ops[op.__name__] = getattr(torch, op.__name__)
            setattr(torch, op.__name__, make_wrapped(getattr(torch, op.__name__), op.__name__))
        for op in self.TENSOR_METHOD_OPS:
            if op.__name__ not in self.original_ops:
                self.original_ops[op.__name__] = getattr(torch.Tensor, op.__name__)
            setattr(torch.Tensor, op.__name__, make_wrapped(getattr(torch.Tensor, op.__name__), op.__name__))

    def wrap_activation_functions(self):
        def make_wrapped(orig_op, func_name):
            def wrapped(*args, **kwargs):
                if SimpleTorchTracer.instance.current_executing_module is None:
                    node_name = SimpleTorchTracer.instance.pre_trace_op(func_name, args, None, *args, **kwargs)
                    output = orig_op(*args, **kwargs)
                    return SimpleTorchTracer.instance.trace_op(node_name, output)
                else:
                    return orig_op(*args, **kwargs)
            return wrapped

        for func in self.ACTIVATION_FUNCTIONS:
            if func.__name__ not in self.original_ops:
                self.original_ops[func.__name__] = func
            setattr(F, func.__name__, make_wrapped(func, func.__name__))

    def restore_activation_functions(self):
        for func_name, original_func in self.original_ops.items():
            if hasattr(F, func_name):
                setattr(F, func_name, original_func)

    def restore_tensor_ops(self):
        for op in self.TENSOR_FUNCTION_OPS:
            setattr(torch, op.__name__, self.original_ops[op.__name__])
        for op in self.TENSOR_METHOD_OPS:
            setattr(torch.Tensor, op.__name__, self.original_ops[op.__name__])

    def trace_model(self, model, input_tensor):
        SimpleTorchTracer.instance = self
        self.wrap_tensor_ops()
        self.wrap_activation_functions()
        
        self.traverse_model(model)

        input_tensor._tl_name = 'input'
        self.adj_list['input'] = {
            'edges': [],
            'input_dims': tuple(input_tensor.shape),
            'output_dims': tuple(input_tensor.shape),
            'failed': False,
            'is_module': False,
        }
        self.node_to_base_name_map['input'] = 'input'

        exception = None
        try:
            with torch.no_grad():
                output = model(input_tensor)

                output_node_name = 'output'
                self.adj_list[output_node_name] = {
                    'edges': [],
                    'input_dims': self.format_dims(self.adj_list[self.last_successful_op]['output_dims']),
                    'output_dims': self.format_dims(tuple(output.shape) if isinstance(output, torch.Tensor) else output),
                    'failed': False,
                    'is_module': False,
                }
                self.node_to_base_name_map[output_node_name] = output_node_name
                
                self.adj_list[self.last_successful_op]['edges'].append(output_node_name)
        except Exception as e:
            exception = e
        finally:
            self.restore_tensor_ops()
            self.restore_activation_functions()

        plot_graph(self.adj_list, self.node_to_base_name_map, self.module_info, self.func_info_map, self.parent_module_to_nodes, self.parent_module_to_depth)

        if exception is not None:
            raise exception

