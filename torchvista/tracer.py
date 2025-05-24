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

MODULES = get_all_nn_modules() - CONTAINER_MODULES


def process_graph(model, inputs, adj_list, node_to_base_name_map, module_info, func_info_map, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, node_to_ancestors):
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
    traced_modules = set()
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
        nonlocal op_type_counters, module_to_node_name, module_info, node_to_base_name_map, module_reuse_count
        if module:
            if module not in module_to_node_name:
                op_type_counters[op_type] += 1
                base_name = f"{op_type}_{op_type_counters[op_type]}"
                module_to_node_name[module] = base_name
                module_info[base_name] = get_module_info(module)
                node_to_base_name_map[base_name] = base_name
                return base_name, NodeType.MODULE.value
            else:
                base_name = module_to_node_name[module]
                module_reuse_count[base_name] = module_reuse_count.get(base_name, 0) + 1
                reused_name = f"{base_name}_{module_reuse_count[base_name]}"
                node_to_base_name_map[reused_name] = base_name
                return reused_name, NodeType.MODULE.value
        else:
            op_type_counters[op_type] += 1
            op_name = f"{op_type}_{op_type_counters[op_type]}"
            node_to_base_name_map[op_name] = op_name
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
        # This can happen in some discovered operations which don't take any inputs. For these, we don't
        # have to put nodes in the graph.
        if not inputs:
            return
        
        nonlocal current_op, last_successful_op, last_tensor_input_id, last_np_array_input_id, last_numeric_input_id
        op_name, node_type = get_unique_op_name(op_type, module)
        
        graph_node_name_to_without_suffix[op_name] = op_type
        adj_list[op_name] = {
            'edges': [],
            'failed': True,
            'node_type': node_type,
        }
        input_tensors = extract_tensors_from_obj(inputs)
        
        for inp in input_tensors:
            if hasattr(inp, '_tensor_source_name'):
                dims = format_dims(tuple(inp.shape))
                adj_list[inp._tensor_source_name]['edges'].append({'target': op_name, 'dims': dims})
            elif isinstance(inp, torch.Tensor):
                dims = format_dims(tuple(inp.shape))
                adj_list[f'tensor_{last_tensor_input_id}'] = {
                    'edges': [],
                    'failed': False,
                    'node_type': 'Constant',
                }
                adj_list[f'tensor_{last_tensor_input_id}']['edges'].append({'target': op_name, 'dims': dims})
                node_to_ancestors[f'tensor_{last_tensor_input_id}'] = module_stack[::-1]
                constant_node_names.append(f'tensor_{last_tensor_input_id}')
                last_tensor_input_id += 1

        for inp in inputs:
            if isinstance(inp, np.ndarray):
                dims = format_dims(tuple(inp.shape))
                adj_list[f'np_array_{last_np_array_input_id}'] = {
                    'edges': [],
                    'failed': False,
                    'node_type': NodeType.CONSTANT.value,
                }
                adj_list[f'np_array_{last_np_array_input_id}']['edges'].append({'target': op_name, 'dims': dims})
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
        def make_wrapped(orig_func, func_name, namespace):
            def wrapped(*args, **kwargs):
                nonlocal current_executing_module, current_executing_function
                if current_executing_module is None and current_executing_function is None:
                    current_executing_function = func_name
                    node_name = pre_trace_op(func_name, args, None, *args, **kwargs)
                    output = orig_func(*args, **kwargs)
                    current_executing_function = None
                    output = trace_op(node_name, output)
                    return output
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
            elif namespace == 'torch.nn.init':
                module = torch.nn.init
            elif namespace == 'torch.linalg':
                module = torch.linalg
            else:
                continue

            try:
                orig_func = getattr(module, func_name)
                if callable(orig_func):
                    original_ops[(namespace, func_name)] = orig_func
            except AttributeError:
                pass

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
            elif namespace == 'torch.nn.init':
                module = torch.nn.init
            elif namespace == 'torch.linalg':
                module = torch.linalg
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
            node_to_base_name_map[input_name] = input_name
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
        
                        node_to_base_name_map[output_node_name] = output_node_name
                        output_node_set.add(output_node_name)
        
                    # Always create the edge, pointing to the *correct* output node
                    dims = format_dims(tuple(output_tensor.shape))
                    target_node_name = seen_tensors[tensor_id]
                    adj_list[output_tensor._tensor_source_name]['edges'].append({
                        'target': target_node_name,
                        'dims': dims
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
    

def plot_graph(adj_list, module_name_to_base_name, module_info, func_info_map, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, ancestor_map, max_module_expansion_depth):
    unique_id = str(uuid.uuid4())
    template_str = resources.read_text('torchvista.templates', 'graph.html')
    d3_source = resources.read_text('torchvista.assets', 'd3.min.js')
    d3_source = resources.read_text('torchvista.assets', 'd3.min.js')
    viz_source = resources.read_text('torchvista.assets', 'viz-standalone.js')

    template = Template(template_str)
        
    output = template.safe_substitute({
        'adj_list_json': json.dumps(adj_list),
        'module_info_json': json.dumps(module_info),
        'func_info_map_json': json.dumps(func_info_map),
        'module_name_to_base_name_json': json.dumps(module_name_to_base_name),
        'parent_module_to_nodes_json': json.dumps(parent_module_to_nodes),
        'parent_module_to_depth_json': json.dumps(parent_module_to_depth),
        'graph_node_name_to_without_suffix': json.dumps(graph_node_name_to_without_suffix),
        'ancestor_map': json.dumps(ancestor_map),
        'unique_id': unique_id,
        'd3_source': d3_source,
        'viz_source': viz_source,
        'max_module_expansion_depth': max_module_expansion_depth,
    })
    display(HTML(output))


def _get_demo_html_str(model, inputs, code_contents, max_module_expansion_depth=3):
    max_module_expansion_depth = max(max_module_expansion_depth, 0)
    adj_list = {}
    module_info = {}
    func_info_map = {}
    node_to_base_name_map = {}
    parent_module_to_nodes = defaultdict(list)
    parent_module_to_depth = {}
    graph_node_name_to_without_suffix = {}
    node_to_ancestors = defaultdict(list)

    exception = None

    try:
        process_graph(model, inputs, adj_list, node_to_base_name_map, module_info, func_info_map, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, node_to_ancestors)
    except Exception as e:
        exception = e

    unique_id = str(uuid.uuid4())
    template_str = resources.read_text('torchvista.templates', 'demo-graph.html')
    d3_source = resources.read_text('torchvista.assets', 'd3.min.js')
    d3_source = resources.read_text('torchvista.assets', 'd3.min.js')
    viz_source = resources.read_text('torchvista.assets', 'viz-standalone.js')

    template = Template(template_str)
        
    output = template.safe_substitute({
        'adj_list_json': json.dumps(adj_list),
        'module_info_json': json.dumps(module_info),
        'func_info_map_json': json.dumps(func_info_map),
        'module_name_to_base_name_json': json.dumps(node_to_base_name_map),
        'parent_module_to_nodes_json': json.dumps(parent_module_to_nodes),
        'parent_module_to_depth_json': json.dumps(parent_module_to_depth),
        'graph_node_name_to_without_suffix': json.dumps(graph_node_name_to_without_suffix),
        'ancestor_map': json.dumps(build_immediate_ancestor_map(node_to_ancestors, adj_list)),
        'unique_id': unique_id,
        'd3_source': d3_source,
        'viz_source': viz_source,
        'code_contents': code_contents,
        'error_contents': str(exception) if exception else "",
        'max_module_expansion_depth': max_module_expansion_depth,
    })
    return output, exception


def trace_model(model, inputs, max_module_expansion_depth=3):
    adj_list = {}
    module_info = {}
    func_info_map = {}
    node_to_base_name_map = {}
    parent_module_to_nodes = defaultdict(list)
    parent_module_to_depth = {}
    graph_node_name_to_without_suffix = {}
    node_to_ancestors = defaultdict(list)
    max_module_expansion_depth = max(max_module_expansion_depth, 0)

    exception = None

    try:
        process_graph(model, inputs, adj_list, node_to_base_name_map, module_info, func_info_map, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, node_to_ancestors)
    except Exception as e:
        exception = e

    plot_graph(adj_list, node_to_base_name_map, module_info, func_info_map, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, build_immediate_ancestor_map(node_to_ancestors, adj_list), max_module_expansion_depth)


    if exception is not None:
        raise exception
