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

import torch
import torch.nn as nn

import json
from IPython.display import display, HTML
import json
from pathlib import Path
from string import Template
import uuid
from collections import defaultdict


def get_all_nn_modules():
    import inspect
    import pkgutil
    import importlib
    import torch.nn as nn

    try:
        import torchvision
    except ImportError:
        torchvision = None
    
    try:
        import torchaudio
    except ImportError:
        torchaudio = None
    
    try:
        import torchtext
    except ImportError:
        torchtext = None

    modules_to_scan = [nn, torchvision, torchaudio, torchtext]

    visited = set()
    module_classes = set()

    def walk_module(mod):
        if mod in visited:
            return
        visited.add(mod)

        try:
            for name, obj in inspect.getmembers(mod):
                if inspect.isclass(obj) and issubclass(obj, nn.Module):
                    module_classes.add(obj)
        except Exception:
            return  # Skip modules that can't be introspected

        # Recursively explore submodules
        if hasattr(mod, '__path__'):
            for _, subname, ispkg in pkgutil.iter_modules(mod.__path__, mod.__name__ + "."):
                try:
                    submod = importlib.import_module(subname)
                    walk_module(submod)
                except Exception:
                    continue  # skip if can't import

    for mod in modules_to_scan:
        if mod is not None:
            walk_module(mod)

    return module_classes


CONTAINER_MODULES = []

MODULES = get_all_nn_modules() - CONTAINER_MODULES

FUNCTIONS = [
    {'namespace': 'torch', 'function': 'chunk'},
    {'namespace': 'torch', 'function': 'split'},
    {'namespace': 'torch', 'function': 'stack'},
    {'namespace': 'torch', 'function': 'matmul'},
    {'namespace': 'torch', 'function': 'lobpcg'},
    {'namespace': 'torch', 'function': 'sym_not'},
    {'namespace': 'torch', 'function': 'unravel_index'},
    {'namespace': 'torch', 'function': 'sym_int'},
    {'namespace': 'torch', 'function': 'sym_float'},
    {'namespace': 'torch', 'function': 'sym_max'},
]

def trace_model(model, input_tensor):
    adj_list = {}
    op_type_counters = defaultdict(int)
    last_successful_op = None
    module_to_node_name = {}
    current_op = None
    original_ops = {}
    module_reuse_count = {}
    module_info = {}
    func_info_map = {}
    node_to_base_name_map = {}
    module_hierarchy = {}
    traced_modules = set()
    current_executing_module = None
    current_executing_function = None
    module_stack = []
    parent_module_to_nodes = defaultdict(list)
    parent_module_to_depth = {}
    original_module_forwards = {}
    last_tensor_input_id = 0
    last_primitive_input_id = 0

    def format_dims(dims):
        if isinstance(dims, tuple):
            return f"({', '.join(map(str, dims))})"
        elif isinstance(dims, list):
            return f"[{', '.join(format_dims(d) for d in dims)}]"
        else:
            return str(dims)

    def get_unique_op_name(op_type, module=None):
        nonlocal op_type_counters, module_to_node_name, module_info, node_to_base_name_map, module_reuse_count
        if module:
            if module not in module_to_node_name:
                op_type_counters[op_type] += 1
                base_name = f"{op_type}_{op_type_counters[op_type]}"
                module_to_node_name[module] = base_name
                module_info[base_name] = get_module_info(module)
                node_to_base_name_map[base_name] = base_name
                return base_name, True
            else:
                base_name = module_to_node_name[module]
                module_reuse_count[base_name] = module_reuse_count.get(base_name, 0) + 1
                reused_name = f"{base_name}_Reused_{module_reuse_count[base_name]}"
                node_to_base_name_map[reused_name] = base_name
                return reused_name, True
        else:
            op_type_counters[op_type] += 1
            op_name = f"{op_type}_{op_type_counters[op_type]}"
            node_to_base_name_map[op_name] = op_name
            return op_name, False

    def get_module_info(module):
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

    def format_arg(arg, max_length=50):
        if isinstance(arg, torch.Tensor):
            return f"tensor({','.join(str(d) for d in arg.shape)})"
        elif isinstance(arg, (list, tuple)) and all(isinstance(item, torch.Tensor) for item in arg):
            return str([format_arg(item) for item in arg])
        elif isinstance(arg, (int, float, bool)):
            return arg
        else:
            arg_str = str(arg)
            if len(arg_str) > max_length:
                return arg_str[:max_length - 3] + "..."
            return arg_str

    def capture_args(*args, **kwargs):
        formatted_args = [format_arg(arg) for arg in args]
        formatted_kwargs = {k: format_arg(v) for k, v in kwargs.items()}
        return formatted_args, formatted_kwargs

    def pre_trace_op(op_type, inputs, module=None, *args, **kwargs):
        nonlocal current_op, last_successful_op, last_primitive_input_id, last_tensor_input_id
        op_name, is_module = get_unique_op_name(op_type, module)
        
        input_dims = tuple(inputs[0].shape) if isinstance(inputs[0], torch.Tensor) else \
                     [tuple(t.shape) for t in inputs[0]] if isinstance(inputs[0], (list, tuple)) and all(isinstance(t, torch.Tensor) for t in inputs[0]) \
                     else inputs[0]

        adj_list[op_name] = {
            'edges': [],
            'input_dims': format_dims(input_dims),
            'output_dims': None,
            'failed': True,
            'is_module': is_module,
        }

        for inp in inputs:
            if isinstance(inp, torch.Tensor) and hasattr(inp, '_tensor_source_name'):
                adj_list[inp._tensor_source_name]['edges'].append(op_name)
            elif isinstance(inp, (list, tuple)):
                for t in inp:
                    if isinstance(t, torch.Tensor) and hasattr(t, '_tensor_source_name'):
                        adj_list[t._tensor_source_name]['edges'].append(op_name)
            elif isinstance(inp, torch.Tensor):
                adj_list[f'tensor_{last_tensor_input_id}'] = {
                    'edges': [op_name],
                    'input_dims': format_dims(inp.shape),
                    'output_dims': format_dims(inp.shape),
                    'failed': False,
                    'is_module': False,
                }
                last_tensor_input_id += 1
            else:
                adj_list[f'{type(inp).__name__}_{last_primitive_input_id}'] = {
                    'edges': [op_name],
                    'input_dims': '',
                    'output_dims': '',
                    'failed': False,
                    'is_module': False,
                }
                last_primitive_input_id += 1
                
                

        formatted_args, formatted_kwargs = capture_args(*args, **kwargs)
        func_info_map[op_name] = {
            "positional_args": formatted_args,
            "keyword_args": formatted_kwargs
        }

        current_op = op_name

        depth = 1
        for parent in module_stack[::-1]:
            parent_module_to_nodes[parent].append(op_name)
            parent_module_to_depth[parent] = max(depth, 0 if parent not in parent_module_to_depth else parent_module_to_depth[parent])
            depth += 1

        return op_name

    def trace_op(op_name, output):
        nonlocal last_successful_op, current_op
        output_dims = format_dims(tuple(output.shape) if isinstance(output, torch.Tensor) else output)
        adj_list[op_name]['output_dims'] = output_dims
        adj_list[op_name]['failed'] = False

        if isinstance(output, torch.Tensor):
            output._tensor_source_name = op_name

        last_successful_op = op_name
        current_op = None

        return output

    def wrap_traced_module(module):
        nonlocal current_executing_module
        if module in original_module_forwards:
            return
        orig_forward = module.forward
        original_module_forwards[module] = orig_forward

        def wrapped_forward(*args, **kwargs):
            nonlocal current_executing_module
            current_executing_module = module
            op_name = pre_trace_op(type(module).__name__, args, module, *args, **kwargs)
            output = orig_forward(*args, **kwargs)
            result = trace_op(op_name, output)
            current_executing_module = None
            return result

        module.forward = wrapped_forward
        traced_modules.add(module)

    def wrap_untraced_module(module):
        if module in original_module_forwards:
            return
        orig_forward = module.forward
        original_module_forwards[module] = orig_forward

        def wrapped_forward(*args, **kwargs):
            module_stack.append(get_unique_op_name(type(module).__name__, module)[0])
            output = orig_forward(*args, **kwargs)
            module_stack.pop()
            return output

        module.forward = wrapped_forward
        traced_modules.add(module)

    def traverse_model(model, parent=None):
        for name, module in model.named_children():
            module_hierarchy[module] = parent
            if type(module) in MODULES:
                wrap_traced_module(module)
            else:
                wrap_untraced_module(module)
                if list(model.named_children()):
                    traverse_model(module, parent=module)

    def wrap_functions():
        def make_wrapped(orig_func, func_name):
            def wrapped(*args, **kwargs):
                nonlocal current_executing_module, current_executing_function
                if current_executing_module is None and current_executing_function is None:
                    current_executing_function = func_name
                    node_name = pre_trace_op(func_name, args, None, *args, **kwargs)
                    output = orig_func(*args, **kwargs)
                    current_executing_function = None
                    return trace_op(node_name, output)
                else:
                    return orig_func(*args, **kwargs)
            return wrapped

        for func_info in FUNCTIONS:
            namespace = func_info['namespace']
            func_name = func_info['function']
            
            if namespace == 'torch':
                module = torch
            elif namespace == 'torch.functional':
                module = torch.functional
            elif namespace == 'torch.Tensor':
                module = torch.Tensor
            elif namespace == 'torch.nn.functional':
                module = torch.nn.functional
            else:
                continue

            try:
                orig_func = getattr(module, func_name)
                if callable(orig_func):
                    wrapped_func = make_wrapped(orig_func, func_name)
                    original_ops[(namespace, func_name)] = orig_func
                    setattr(module, func_name, wrapped_func)
            except AttributeError:
                pass

    def restore_functions():
        for (namespace, func_name), orig_func in original_ops.items():
            if namespace == 'torch':
                module = torch
            elif namespace == 'torch.functional':
                module = torch.functional
            elif namespace == 'torch.Tensor':
                module = torch.Tensor
            elif namespace == 'torch.nn.functional':
                module = torch.nn.functional
            else:
                continue

            setattr(module, func_name, orig_func)

    def restore_modules():
        for module, original_call in original_module_forwards.items():
            module.forward = original_call
        traced_modules.clear()

    def cleanup_tensor_attributes(obj):
        if isinstance(obj, torch.Tensor):
            if hasattr(obj, '_tensor_source_name'):
                delattr(obj, '_tensor_source_name')
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                cleanup_tensor_attributes(item)
        elif isinstance(obj, dict):
            for value in obj.values():
                cleanup_tensor_attributes(value)

    try:
        wrap_functions()
        traverse_model(model)

        input_tensor._tensor_source_name = 'input'
        adj_list['input'] = {
            'edges': [],
            'input_dims': tuple(input_tensor.shape),
            'output_dims': tuple(input_tensor.shape),
            'failed': False,
            'is_module': False,
        }
        node_to_base_name_map['input'] = 'input'

        exception = None
        with torch.no_grad():
            output = model(input_tensor)

            cleanup_tensor_attributes(output)

            output_node_name = 'output'
            adj_list[output_node_name] = {
                'edges': [],
                'input_dims': format_dims(adj_list[last_successful_op]['output_dims']),
                'output_dims': format_dims(tuple(output.shape) if isinstance(output, torch.Tensor) else output),
                'failed': False,
                'is_module': False,
            }
            node_to_base_name_map[output_node_name] = output_node_name
            
            adj_list[last_successful_op]['edges'].append(output_node_name)
    except Exception as e:
        exception = e
    finally:
        restore_functions()
        restore_modules()
        cleanup_tensor_attributes(input_tensor)

    if exception is not None:
        raise exception

    return adj_list, node_to_base_name_map, module_info, func_info_map, parent_module_to_nodes, parent_module_to_depth

def plot_graph(adj_list, module_name_to_base_name, module_info, tensor_op_info, parent_module_to_nodes, parent_module_to_depth):
    unique_id = str(uuid.uuid4())
    template_path = Path('graph.html')
    with template_path.open('r') as file:
        template_str = file.read()

    template = Template(template_str)
        
    output = template.safe_substitute({
        'adj_list_json': json.dumps(adj_list),
        'module_info_json': json.dumps(module_info),
        'tensor_op_info_json': json.dumps(tensor_op_info),
        'module_name_to_base_name_json': json.dumps(module_name_to_base_name),
        'parent_module_to_nodes_json': json.dumps(parent_module_to_nodes),
        'parent_module_to_depth_json': json.dumps(parent_module_to_depth),
        'unique_id': unique_id,
    })
    display(HTML(output))
