import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torch.overrides import get_ignored_functions
from pathlib import Path
from string import Template
import uuid
from collections import defaultdict
from .overrides import CONTAINER_MODULES, FUNCTIONS
import warnings

import json
from IPython.display import display, HTML
import numpy as np
import numbers
from enum import Enum
from importlib import resources


class NodeType(Enum):
    MODULE = "Module"
    OPERATION = "Operation"
    INPUT = "Input"
    OUTPUT = "Output"
    CONSTANT = "Constant"


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
    except Exception:
        print('[warning] torchaudio available, but import failed and hence torchvista cannot trace torchaudio operations.\
               If you need torchaudio tracing, run `import torchaudio` separately to debug what is wrong.')
        torchaudio = None
    
    try:
        import torchtext
    except ImportError:
        torchtext = None
    except Exception:
        print('[warning] torchtext available, but import failed and hence torchvista cannot trace torchtext operations.\
               If you need torchtext tracing, run `import torchtext` separately to debug what is wrong.')
        torchtext = None

    modules_to_scan = [nn, torchvision, torchaudio, torchtext]

    visited = set()
    module_classes = set()

    def walk_module(mod):
        if mod in visited:
            return
        visited.add(mod)

        try:
            for _, obj in inspect.getmembers(mod):
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

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    MODULES = get_all_nn_modules() - CONTAINER_MODULES


def process_graph(model, inputs, adj_list, module_info, func_info, node_to_module_path, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, node_to_ancestors, show_non_gradient_nodes, forced_module_tracing_depth):
    last_successful_op = None
    current_op = None
    current_executing_module = None
    current_executing_function = None
    last_tensor_input_id = 0
    last_np_array_input_id = 0
    last_numeric_input_id = 0

    op_type_counters = defaultdict(int)
    module_to_node_name = {}
    original_ops = {}
    module_reuse_count = {}
    module_hierarchy = {}
    wrapped_modules = set()
    module_stack = []
    original_module_forwards = {}
    nodes_to_delete = []
    constant_node_names = []
    output_node_set = set()


    def format_dims(dims):
        def helper():
            if isinstance(dims, tuple):
                return f"({', '.join(map(str, dims))})"
            elif isinstance(dims, list):
                return f"[{', '.join(helper(d) for d in dims)}]"
            else:
                return "()" if str(dims) == "()" else str(dims)
        result = helper()
        return "( )" if result == "()"  else result

    def get_unique_op_name(op_type, module=None):
        nonlocal op_type_counters, module_to_node_name, module_info, module_reuse_count
        if module:
            op_type_counters[op_type] += 1
            base_name = f"{op_type}_{op_type_counters[op_type]}"
            module_to_node_name[module] = base_name
            module_info[base_name] = get_module_info(module)
            return base_name, NodeType.MODULE.value
        else:
            op_type_counters[op_type] += 1
            op_name = f"{op_type}_{op_type_counters[op_type]}"
            return op_name, NodeType.OPERATION.value

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

    def format_arg(arg):
        def _format(value):
            if isinstance(value, torch.Tensor):
                return {
                    "_type": "tensor",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype)
                }
            elif isinstance(value, np.ndarray):
                return {
                    "_type": "ndarray",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype)
                }
            elif isinstance(value, (list, tuple)):
                return [_format(v) for v in value]
            elif isinstance(value, dict):
                return {str(k): _format(v) for k, v in value.items()}
            elif isinstance(value, (int, float, bool, str, type(None))):
                return value
            else:
                return {
                    "_type": type(value).__name__,
                    "repr": str(value)[:50]  # fallback for unknowns
                }

        return _format(arg)

    def capture_args(*args, **kwargs):
        formatted_args = [format_arg(arg) for arg in args]
        formatted_kwargs = {k: format_arg(v) for k, v in kwargs.items()}
        return formatted_args, formatted_kwargs

    def record_op_parameters(op_name, *args, **kwargs):
        formatted_args, formatted_kwargs = capture_args(*args, **kwargs)
        func_info[op_name] = {
            "positional_args": formatted_args,
            "keyword_args": formatted_kwargs
        }

    def pre_trace_op(op_name, node_type, inputs, *args, **kwargs):        
        nonlocal current_op, last_successful_op, last_tensor_input_id, last_np_array_input_id, last_numeric_input_id

        input_tensors = extract_tensors_from_obj(inputs) + extract_tensors_from_obj(args) + extract_tensors_from_obj(kwargs)
        # This can happen in some discovered operations which don't take any inputs. For these, we don't
        # have to put nodes in the graph.
        if len(input_tensors) == 0:
            return
        adj_list[op_name] = {
            'edges': [],
            'failed': True,
            'node_type': node_type,
        }
        
        for inp in input_tensors:
            if hasattr(inp, '_tensor_source_name'):
                dims = format_dims(tuple(inp.shape))
                adj_list[inp._tensor_source_name]['edges'].append({'target': op_name, 'dims': dims, 'edge_data_id': id(inp)})
            elif isinstance(inp, torch.Tensor) and show_non_gradient_nodes:
                dims = format_dims(tuple(inp.shape))
                adj_list[f'tensor_{last_tensor_input_id}'] = {
                    'edges': [],
                    'failed': False,
                    'node_type': 'Constant',
                }
                adj_list[f'tensor_{last_tensor_input_id}']['edges'].append({'target': op_name, 'dims': dims, 'edge_data_id': id(inp)})
                node_to_ancestors[f'tensor_{last_tensor_input_id}'] = module_stack[::-1]
                constant_node_names.append(f'tensor_{last_tensor_input_id}')
                last_tensor_input_id += 1

        if show_non_gradient_nodes:
            for inp in inputs:
                if isinstance(inp, np.ndarray):
                    dims = format_dims(tuple(inp.shape))
                    adj_list[f'np_array_{last_np_array_input_id}'] = {
                        'edges': [],
                        'failed': False,
                        'node_type': NodeType.CONSTANT.value,
                    }
                    adj_list[f'np_array_{last_np_array_input_id}']['edges'].append({'target': op_name, 'dims': dims, 'edge_data_id': id(inp),})
                    constant_node_names.append(f'np_array_{last_np_array_input_id}')
                    node_to_ancestors[f'np_array_{last_np_array_input_id}'] = module_stack[::-1]
                    last_np_array_input_id += 1
                elif isinstance(inp, numbers.Number):
                    dims = "( )"
                    adj_list[f'scalar_{last_numeric_input_id}'] = {
                        'edges': [],
                        'failed': False,
                        'node_type': NodeType.CONSTANT.value,
                    }
                    adj_list[f'scalar_{last_numeric_input_id}']['edges'].append({'target': op_name, 'dims': dims})
                    constant_node_names.append(f'scalar_{last_numeric_input_id}')
                    node_to_ancestors[f'scalar_{last_numeric_input_id}'] = module_stack[::-1]
                    last_numeric_input_id += 1


        record_op_parameters(op_name, *args, **kwargs)

        current_op = op_name

        depth = 1
        for parent in module_stack[::-1]:
            parent_module_to_nodes[parent].append(op_name)
            parent_module_to_depth[parent] = max(depth, 0 if parent not in parent_module_to_depth else parent_module_to_depth[parent])
            depth += 1

        node_to_ancestors[op_name] = module_stack[::-1]

        return op_name

    def extract_tensors_from_obj(obj, max_depth=5, current_depth=0):
        """Recursively extracts all tensors from any object structure.
        
        Args:
            obj: Any object that might contain tensors
            max_depth: Maximum recursion depth to prevent infinite loops
            current_depth: Current recursion depth
            
        Returns:
            List of tensors found in the object
        """
        if obj is None:
            return []
        if current_depth >= max_depth:
            return []
        
        # Base case: object is a tensor
        if isinstance(obj, torch.Tensor):
            return [obj]
        
        # Recursive cases
        tensors = []
        
        # Handle lists, tuples, and other iterables
        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                tensors.extend(extract_tensors_from_obj(item, max_depth, current_depth + 1))
        
        # Handle dictionaries
        elif isinstance(obj, dict):
            for value in obj.values():
                tensors.extend(extract_tensors_from_obj(value, max_depth, current_depth + 1))
        
        # Handle custom objects with accessible attributes
        elif hasattr(obj, '__dict__'):
            for attr_name in dir(obj):
                # Skip private attributes and callable methods
                if attr_name.startswith('_') or callable(getattr(obj, attr_name, None)):
                    continue
                
                try:
                    attr_value = getattr(obj, attr_name)
                    # Avoid problematic attributes like gradients
                    if attr_name in ['grad', 'grad_fn', '_backward_hooks']:
                        continue
                    tensors.extend(extract_tensors_from_obj(attr_value, max_depth, current_depth + 1))
                except:
                    # Skip attributes that cause errors
                    continue
        
        return tensors

    def trace_op(op_name, output):
        # Because some discovered operations don't get added to the adj_list in pre_trace_op
        if op_name not in adj_list:
            return output
        nonlocal last_successful_op, current_op
        last_successful_op = op_name
        current_op = None
        output_tensors = extract_tensors_from_obj(output)

        if not output_tensors:
            # No tensors found in the output
            nodes_to_delete.append(op_name)
            return output
        
        adj_list[op_name]['failed'] = False
        
        # Tag each tensor with the source operation
        for tensor in output_tensors:
            tensor._tensor_source_name = op_name

        # node_to_ancestors[op_name] = module_stack[::-1]

        return output

    def wrap_module(module):
        nonlocal current_executing_module, forced_module_tracing_depth
        if module in original_module_forwards:
            return
        orig_forward = module.forward
        original_module_forwards[module] = orig_forward

        def wrapped_forward(*args, **kwargs):
            nonlocal current_executing_module, forced_module_tracing_depth
            if forced_module_tracing_depth is not None and forced_module_tracing_depth < len(module_stack):
                # This module might have been overriden as a false positive
                # (because it was at a lower depth in the named_children hierarchy)
                return orig_forward(*args, **kwargs)
            is_traced = False
            if forced_module_tracing_depth is None and type(module) in MODULES:
                is_traced = True
            elif forced_module_tracing_depth is not None and forced_module_tracing_depth <= len(module_stack):
                is_traced = True
            if is_traced:
                current_executing_module = module
                module_name, node_type = get_unique_op_name(type(module).__name__, module)
                graph_node_name_to_without_suffix[module_name] = type(module).__name__
                node_to_module_path[module_name] = type(module).__module__
                pre_trace_op(module_name, node_type, args, *args, **kwargs)
                module_stack.append(module_name)
                output = orig_forward(*args, **kwargs)
                module_stack.pop()
                result = trace_op(module_name, output)
                current_executing_module = None
                return result
            else:
                module_name, _ = get_unique_op_name(type(module).__name__, module)
                graph_node_name_to_without_suffix[module_name] = type(module).__name__
                node_to_module_path[module_name] = type(module).__module__
                module_stack.append(module_name)
                record_op_parameters(module_name, *args, **kwargs)
                output = orig_forward(*args, **kwargs)
                module_stack.pop()
                return output

        module.forward = wrapped_forward
        wrapped_modules.add(module)

    def has_forward_method(module):
        return module.__class__.forward is not torch.nn.Module.forward

    def traverse_model(model, depth=0, parent=None):
        for name, module in model.named_children():
            module_hierarchy[module] = parent
            
            if has_forward_method(module):
                # Some modules like ModuleList don't have forward() implemented
                wrap_module(module)

            if (forced_module_tracing_depth is not None and depth < forced_module_tracing_depth) \
                or (forced_module_tracing_depth is None and type(module) not in MODULES) or not has_forward_method(module):
                # This is an approximate control with potential false positives getting traced.
                # But during tracing, the wrapped forward will check the depth and decide whether to actually wrap it or not.
                # Think of a case like
                # class PositionalTransformer(nn.Module):
                # def __init__(self):
                #     super().__init__()
                #     self.pos_embed = nn.Parameter(torch.randn(10, 1, 32))
                #     self.encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4) <- gets passed below to TransformerEncoder
                #     self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
                # 
                if list(module.named_children()):
                    if has_forward_method(module):
                        traverse_model(module, depth=depth+1, parent=module)
                    else:
                        # If the module doesn't have a forward method this doesn't count towards the depth, and we want to traverse its children
                        # This happens to modules like ModuleList.
                        traverse_model(module, depth=depth, parent=module)

    def wrap_functions():
        def make_wrapped(orig_func, func_name, namespace):
            def wrapped(*args, **kwargs):
                nonlocal current_executing_module, current_executing_function
                if current_executing_module is None and current_executing_function is None:
                    current_executing_function = func_name
                    node_to_module_path[func_name] = namespace
                    node_name, node_type = get_unique_op_name(func_name)
                    graph_node_name_to_without_suffix[node_name] = func_name
                    node_to_module_path[node_name] = namespace
                    pre_trace_op(node_name, node_type, args, *args, **kwargs)
                    output = orig_func(*args, **kwargs)
                    current_executing_function = None
                    output = trace_op(node_name, output)
                    return output
                else:
                    return orig_func(*args, **kwargs)
            return wrapped

        for func in FUNCTIONS:
            namespace = func['namespace']
            func_name = func['function']
            
            if namespace == 'torch':
                module = torch
            elif namespace == 'torch.functional':
                module = torch.functional
            elif namespace == 'torch.Tensor':
                module = torch.Tensor
            elif namespace == 'torch.nn.functional':
                module = torch.nn.functional
            elif namespace == 'torch.nn.init':
                module = torch.nn.init
            elif namespace == 'torch.linalg':
                module = torch.linalg
            elif namespace == 'torch.ops.torchvision':
                module = torch.ops.torchvision
            else:
                continue

            try:
                orig_func = getattr(module, func_name)
                if callable(orig_func):
                    original_ops[(namespace, func_name)] = orig_func
            except AttributeError:
                pass

        for func in FUNCTIONS:
            namespace = func['namespace']
            func_name = func['function']
            
            if namespace == 'torch':
                module = torch
            elif namespace == 'torch.functional':
                module = torch.functional
            elif namespace == 'torch.Tensor':
                module = torch.Tensor
            elif namespace == 'torch.nn.functional':
                module = torch.nn.functional
            elif namespace == 'torch.nn.init':
                module = torch.nn.init
            elif namespace == 'torch.linalg':
                module = torch.linalg
            elif namespace == 'torch.ops.torchvision':
                module = torch.ops.torchvision
            else:
                continue

            try:
                orig_func = getattr(module, func_name)
                if callable(orig_func):
                    wrapped_func = make_wrapped(orig_func, func_name, namespace)
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
            elif namespace == 'torch.nn.init':
                module = torch.nn.init
            elif namespace == 'torch.linalg':
                module = torch.linalg
            elif namespace == 'torch.ops.torchvision':
                module = torch.ops.torchvision
            else:
                continue

            setattr(module, func_name, orig_func)

    def restore_modules():
        for module, original_call in original_module_forwards.items():
            module.forward = original_call
        wrapped_modules.clear()

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

    def cleanup_graph(adj_list, nodes_to_delete):
        # Step 0: Remove unwanted nodes and their edges
        for node in nodes_to_delete:
            if node in adj_list:
                del adj_list[node]
            for src_node, node_data in adj_list.items():
                node_data['edges'] = [edge for edge in node_data['edges'] if edge['target'] != node]
    
        # Step a: Identify all input nodes based on node_type
        input_nodes = [node for node, data in adj_list.items() 
                      if data.get('node_type') == NodeType.INPUT.value]
        
        # Step 1: Forward DFS from all input nodes
        forward_reachable = set()
    
        def dfs_forward(node):
            if node in forward_reachable:
                return
            forward_reachable.add(node)
            for edge in adj_list.get(node, {}).get('edges', []):
                dfs_forward(edge['target'])
    
        # Run DFS from each input node
        for input_node in input_nodes:
            dfs_forward(input_node)

        # Step 2: Build reverse adjacency list
        reverse_adj_list = {}
        for node, data in adj_list.items():
            for edge in data.get('edges', []):
                target = edge['target']
                reverse_adj_list.setdefault(target, []).append(node)
    
        # Step 3: Backward DFS from output nodes
        backward_reachable = set()
    
        def dfs_backward(node):
            if node in backward_reachable:
                return
            backward_reachable.add(node)
            for source in reverse_adj_list.get(node, []):
                dfs_backward(source)
    
        for output_node in output_node_set:
            if output_node in adj_list:
                dfs_backward(output_node)
    
        # Step 4: Union of forward and backward reachable sets
        base_set = forward_reachable.union(backward_reachable)
    
        # Step 5: Expand to include ancestors of base set
        expanded_set = set()
    
        def dfs_full_backward(node):
            if node in expanded_set:
                return
            expanded_set.add(node)
            for source in reverse_adj_list.get(node, []):
                dfs_full_backward(source)
    
        for node in base_set:
            dfs_full_backward(node)
    
        # Step 6: Prune graph to only keep expanded set
        for node in list(adj_list.keys()):
            if node not in expanded_set:
                del adj_list[node]
    
        for node_data in adj_list.values():
            node_data['edges'] = [edge for edge in node_data['edges'] if edge['target'] in adj_list]

                
    try:
        wrap_functions()
        traverse_model(model)

        inputs_wrapped = (inputs)
        input_tensors = extract_tensors_from_obj(inputs_wrapped)
        for i, tensor in enumerate(input_tensors):
            input_name = f'input_{i}'
            tensor._tensor_source_name = input_name
            graph_node_name_to_without_suffix[input_name] = input_name
            adj_list[input_name] = {
                'edges': [],
                'failed': False,
                'node_type': NodeType.INPUT.value,
            }
            node_to_ancestors[input_name] = []

        exception = None
        with torch.no_grad():
            output = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
            output_tensors = extract_tensors_from_obj(output)
            if output_tensors:
                output_node_name = 'output'
                graph_node_name_to_without_suffix['output'] = 'output'
                
                seen_tensors = {}
                
                for i, output_tensor in enumerate(output_tensors):
                    tensor_id = id(output_tensor)
        
                    # If we haven't seen this tensor before, create a node
                    if tensor_id not in seen_tensors:
                        output_node_name = f'output_{i}'
                        seen_tensors[tensor_id] = output_node_name
        
                        adj_list[output_node_name] = {
                            'edges': [],
                            'failed': False,
                            'node_type': NodeType.OUTPUT.value,
                        }
        
                        output_node_set.add(output_node_name)
        
                    # Always create the edge, pointing to the *correct* output node
                    dims = format_dims(tuple(output_tensor.shape))
                    target_node_name = seen_tensors[tensor_id]
                    if hasattr(output_tensor, '_tensor_source_name'):
                        adj_list[output_tensor._tensor_source_name]['edges'].append({
                            'target': target_node_name,
                            'dims': dims,
                            'edge_data_id': id(output_tensor),
                        })
        
                for output_tensor in output_tensors:
                    cleanup_tensor_attributes(output)


    except Exception as e:
        exception = e
    finally:
        restore_functions()
        restore_modules()
        for tensor in input_tensors:
            cleanup_tensor_attributes(tensor)

    cleanup_graph(adj_list, nodes_to_delete)
    if exception is not None:
        raise exception

def build_immediate_ancestor_map(ancestor_dict, adj_list):
    immediate_ancestor_map = {}
    for node, ancestors in ancestor_dict.items():
        if ancestors and node in adj_list:
            immediate_ancestor_map[node] = ancestors[0]
            for i in range(len(ancestors) - 1):
                if ancestors[i] not in immediate_ancestor_map:
                    immediate_ancestor_map[ancestors[i]] = ancestors[i + 1]
    return immediate_ancestor_map
    

def plot_graph(adj_list, module_info, func_info, node_to_module_path, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, ancestor_map, collapse_modules_after_depth, height, generate_image):
    unique_id = str(uuid.uuid4())
    template_str = resources.read_text('torchvista.templates', 'graph.html')
    d3_source = resources.read_text('torchvista.assets', 'd3.min.js')
    viz_source = resources.read_text('torchvista.assets', 'viz-standalone.js')
    jsoneditor_css = resources.read_text('torchvista.assets', 'jsoneditor-10.2.0.min.css')
    jsoneditor_source = resources.read_text('torchvista.assets', 'jsoneditor-10.2.0.min.js')

    template = Template(template_str)
        
    output = template.safe_substitute({
        'adj_list_json': json.dumps(adj_list),
        'module_info_json': json.dumps(module_info),
        'func_info_json': json.dumps(func_info),
        'parent_module_to_nodes_json': json.dumps(parent_module_to_nodes),
        'parent_module_to_depth_json': json.dumps(parent_module_to_depth),
        'graph_node_name_to_without_suffix': json.dumps(graph_node_name_to_without_suffix),
        'ancestor_map': json.dumps(ancestor_map),
        'unique_id': unique_id,
        'd3_source': d3_source,
        'viz_source': viz_source,
        'jsoneditor_css': jsoneditor_css,
        'jsoneditor_source': jsoneditor_source,
        'collapse_modules_after_depth': collapse_modules_after_depth,
        'node_to_module_path': node_to_module_path,
        'height': height if not generate_image else 0,
        'generate_image': 'true' if generate_image else 'false',
    })
    
    return output


def _get_demo_html_str(model, inputs, code_contents, collapse_modules_after_depth=1, show_non_gradient_nodes=True, forced_module_tracing_depth=None):
    collapse_modules_after_depth = max(collapse_modules_after_depth, 0)
    adj_list = {}
    module_info = {}
    func_info = {}
    parent_module_to_nodes = defaultdict(list)
    parent_module_to_depth = {}
    graph_node_name_to_without_suffix = {}
    node_to_module_path = {}
    node_to_ancestors = defaultdict(list)

    exception = None

    try:
        process_graph(model, inputs, adj_list, module_info, func_info, node_to_module_path, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, node_to_ancestors, show_non_gradient_nodes=show_non_gradient_nodes, forced_module_tracing_depth=forced_module_tracing_depth)
    except Exception as e:
        exception = e

    unique_id = str(uuid.uuid4())
    template_str = resources.read_text('torchvista.templates', 'demo-graph.html')
    d3_source = resources.read_text('torchvista.assets', 'd3.min.js')
    viz_source = resources.read_text('torchvista.assets', 'viz-standalone.js')
    jsoneditor_css = resources.read_text('torchvista.assets', 'jsoneditor-10.2.0.min.css')
    jsoneditor_source = resources.read_text('torchvista.assets', 'jsoneditor-10.2.0.min.js')

    template = Template(template_str)
        
    output = template.safe_substitute({
        'adj_list_json': json.dumps(adj_list),
        'module_info_json': json.dumps(module_info),
        'func_info_json': json.dumps(func_info),
        'parent_module_to_nodes_json': json.dumps(parent_module_to_nodes),
        'parent_module_to_depth_json': json.dumps(parent_module_to_depth),
        'graph_node_name_to_without_suffix': json.dumps(graph_node_name_to_without_suffix),
        'ancestor_map': json.dumps(build_immediate_ancestor_map(node_to_ancestors, adj_list)),
        'unique_id': unique_id,
        'd3_source': d3_source,
        'viz_source': viz_source,
        'code_contents': code_contents,
        'error_contents': str(exception) if exception else "",
        'jsoneditor_css': jsoneditor_css,
        'jsoneditor_source': jsoneditor_source,
        'collapse_modules_after_depth': collapse_modules_after_depth,
        'node_to_module_path': node_to_module_path,
    })
    return output, exception


def trace_model(model, inputs, max_module_expansion_depth=None, show_non_gradient_nodes=True, collapse_modules_after_depth=1, forced_module_tracing_depth=None, height=800, generate_image=False):
    adj_list = {}
    module_info = {}
    func_info = {}
    parent_module_to_nodes = defaultdict(list)
    parent_module_to_depth = {}
    graph_node_name_to_without_suffix = {}
    node_to_module_path = {}
    node_to_ancestors = defaultdict(list)
    if max_module_expansion_depth is not None:
        # hacky backwards compatibility
        collapse_modules_after_depth = max_module_expansion_depth
    collapse_modules_after_depth = max(collapse_modules_after_depth, 0)

    exception = None

    try:
        process_graph(model, inputs, adj_list, module_info, func_info, node_to_module_path, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, node_to_ancestors, show_non_gradient_nodes=show_non_gradient_nodes, forced_module_tracing_depth=forced_module_tracing_depth)
    except Exception as e:
        exception = e
        
    # ---------------------------------------------
    from flask import Flask, render_template_string
    import threading

    def _start_flask_server(html, port=6006):
        app = Flask(__name__)

        @app.route("/")
        def index():
            return render_template_string(html)

        threading.Thread(
            target=lambda: app.run(port=port, debug=False, use_reloader=False),
            daemon=True
        ).start()
        print(f"\n🚀 TorchVista visualization is available at: http://localhost:{port}\n")

    # HTML を生成
    html = plot_graph(adj_list, module_info, func_info, node_to_module_path, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, build_immediate_ancestor_map(node_to_ancestors, adj_list), collapse_modules_after_depth, height, generate_image)
    _start_flask_server(html, port=6006)
    
    input("🔗 Press Enter to terminate the server and exit.")
    # ---------------------------------------------

    if exception is not None:
        raise exception
