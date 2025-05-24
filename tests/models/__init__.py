import importlib
import os

models = {}

for filename in os.listdir(os.path.dirname(__file__)):
    if filename.endswith(".py") and filename != "__init__.py":
        modname = filename[:-3]
        module = importlib.import_module(f".{modname}", package=__name__)
        models[modname] = {
            "model": getattr(module, "model"),
            "example_input": getattr(module, "example_input"),
            "code_contents": getattr(module, "code_contents"),
            "error_contents": getattr(module, "error_contents", ""),
            "max_module_expansion_depth": getattr(module, "max_module_expansion_depth", -1)
        }
