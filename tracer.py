import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import inspect
# from IPython.core.ultratb import AutoFormattedTB
from torch.overrides import get_ignored_functions
import json
# from IPython.display import display, HTML
import json
from pathlib import Path
from string import Template
import uuid
from collections import defaultdict
import types
from overrides import CONTAINER_MODULES, FUNCTIONS
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import inspect
from IPython.core.ultratb import AutoFormattedTB

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

MODULES = get_all_nn_modules() - CONTAINER_MODULES

def plot_graph(adj_list, module_name_to_base_name, module_info, tensor_op_info, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix):
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
        'graph_node_name_to_without_suffix': json.dumps(graph_node_name_to_without_suffix),
        'unique_id': unique_id,
    })
    display(HTML(output))


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
    graph_node_name_to_without_suffix = {}
    node_to_ancestors = defaultdict(list)
    last_tensor_input_id = 0
    last_primitive_input_id = 0

    nodes_to_delete = []

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
                reused_name = f"{base_name} ({module_reuse_count[base_name]})"
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
        # This can happen in some discovered operations which don't take any inputs. For these, we don't
        # have to put nodes in the graph.
        if not inputs:
            return
        
        nonlocal current_op, last_successful_op, last_primitive_input_id, last_tensor_input_id
        op_name, is_module = get_unique_op_name(op_type, module)
        
        input_dims = tuple(inputs[0].shape) if isinstance(inputs[0], torch.Tensor) else \
                     [tuple(t.shape) for t in inputs[0]] if isinstance(inputs[0], (list, tuple)) and all(isinstance(t, torch.Tensor) for t in inputs[0]) \
                     else inputs[0]

        graph_node_name_to_without_suffix[op_name] = op_type
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
                graph_node_name_to_without_suffix[f'tensor_{last_tensor_input_id}'] = 'tensor'
                adj_list[f'tensor_{last_tensor_input_id}'] = {
                    'edges': [op_name],
                    'input_dims': format_dims(inp.shape),
                    'output_dims': format_dims(inp.shape),
                    'failed': False,
                    'is_module': False,
                }
                last_tensor_input_id += 1
            else:
                graph_node_name_to_without_suffix[f'{type(inp).__name__}_{last_primitive_input_id}'] = f'{type(inp).__name__}'
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

        # If we found any tensors in the output, keep the node
        # Try to format the output dimensions in a meaningful way
        if isinstance(output, torch.Tensor):
            output_dims = format_dims(tuple(output.shape))
        elif all(isinstance(t, torch.Tensor) for t in output_tensors):
            output_dims = [format_dims(tuple(t.shape)) for t in output_tensors]
        else:
            output_dims = "complex output with tensors"
        
        adj_list[op_name]['output_dims'] = output_dims
        adj_list[op_name]['failed'] = False
        
        # Tag each tensor with the source operation
        for tensor in output_tensors:
            tensor._tensor_source_name = op_name
                
        output_dims = format_dims(tuple(output.shape) if isinstance(output, torch.Tensor) else output)
        adj_list[op_name]['output_dims'] = output_dims
        adj_list[op_name]['failed'] = False

        node_to_ancestors[op_name] = module_stack[::-1]

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
        for node in nodes_to_delete:
            del adj_list[node]
            for _, l in adj_list.items():
                if node in l['edges']:
                    l['edges'].remove(node)

        reachable = set()
    
        def dfs(node):
            if node in reachable:
                return
            reachable.add(node)
            for neighbor in adj_list.get(node, {}).get('edges', []):
                dfs(neighbor)
    
        dfs('input')
    
        for node in list(adj_list.keys()):
            if node not in reachable:
                del adj_list[node]

    def transform_to_nested(adjacency, ancestry):
        # utility function to get the element in the list before the target, if one is present
        def get_element_before(lst, target):
            try:
                index = lst.index(target)
                return lst[index - 1] if index - 1 >= 0 else None
            except ValueError:
                return None
    
        # Finds the lowest common ancestor between 2 paths, assuming that the paths
        # are ordered from bottom to top
        def find_lca(path1, path2):
            lca = None
            for a, b in zip(path1[::-1],  path2[::-1]):
                if a == b:
                    lca = a
                else:
                    break
            return lca
    
        # Given two nodes, it adds a link between the correct pair of "representative" nodes
        # in the newly constructed nested graph
    
        def add_link(node1, node2, nodes):
            ancestry1, ancestry2 = ancestry[node1], ancestry[node2]
            # Special cases when the LCA cannot be found
            if not ancestry1 and not ancestry2:
                nodes[node1]['edges'].append(node2)
            elif not ancestry1:
                nodes[node1]['edges'].append(ancestry2[-1])
            elif not ancestry2:
                nodes[ancestry1[-1]]['edges'].append(node2)
            else:
                # When LCA is likely to be present
                lca = find_lca(ancestry1, ancestry2)
                if not lca:
                    # This can happen if the 2 nodes have completely disjoint hierarchy paths
                    nodes[ancestry1[-1]]['edges'].append(ancestry2[-1])
                    return
        
                # The node just below the LCA in each node serves as the "representative" node of that node in the newly built graph
                representative_node1 = get_element_before(ancestry1, lca)
                representative_node2 = get_element_before(ancestry2, lca)
                # If the two nodes are in the same subtree at the same level, they
                # will act as their own representative nodes
                representative_node1 = node1 if representative_node1 is None else representative_node1
                representative_node2 = node2 if representative_node2 is None else representative_node2
                nodes[representative_node1]['edges'].append(representative_node2)
    
    
        # Create the basic object (dict) for each node:
        nodes = { 
            subgraph: { 'edges': [], 'subgraphs': {} } 
                for node, subgraphs in ancestry.items()
                for subgraph in (node, *subgraphs)
        }
        # populate the "edges" attributes between basic nodes (or their "representative" nodes)
        for node, val in adjacency.items():
            children = val['edges']
            for child in children:
                add_link(node, child, nodes)
        
        # keep track of the nodes that are to stay at the root level
        root = dict(nodes)
        # populate the "subgraphs" attributes
        for node, ancestors in ancestry.items():
            for child, parent in zip((node, *ancestors), ancestors):
                nodes[parent]['subgraphs'][child] = nodes[child]
                root.pop(child, None)
    
        return root
    
        
    try:
        wrap_functions()
        traverse_model(model)

        input_tensor._tensor_source_name = 'input'
        graph_node_name_to_without_suffix['input'] = 'input'
        adj_list['input'] = {
            'edges': [],
            'input_dims': tuple(input_tensor.shape),
            'output_dims': format_dims(tuple(input_tensor.shape)),
            'failed': False,
            'is_module': False,
        }
        node_to_base_name_map['input'] = 'input'
        node_to_ancestors['input'] = []
        node_to_ancestors['output'] = []

        exception = None
        with torch.no_grad():
            output = model(input_tensor)
            output_tensors = extract_tensors_from_obj(output)
            if output_tensors:
                output_node_name = 'output'
                graph_node_name_to_without_suffix['output'] = 'output'
                
                adj_list[output_node_name] = {
                    'edges': [],
                    'input_dims': "",
                    'output_dims': "",
                    'failed': False,
                    'is_module': False,
                }
                node_to_base_name_map[output_node_name] = output_node_name

                for output_tensor in output_tensors:
                    adj_list[output_tensor._tensor_source_name]['edges'].append(output_node_name)
                for output_tensor in output_tensors:
                    cleanup_tensor_attributes(output)

    except Exception as e:
        exception = e
    finally:
        restore_functions()
        restore_modules()
        cleanup_tensor_attributes(input_tensor)

    cleanup_graph(adj_list, nodes_to_delete)
    plot_graph(adj_list, node_to_base_name_map, module_info, func_info_map, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix)
    # adj_list = transform_to_nested(adj_list, node_to_ancestors)
    if exception is not None:
        raise exception
