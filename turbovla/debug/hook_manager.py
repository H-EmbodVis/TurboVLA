from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from .trace_context import TraceContext

def install_leaf_hooks(model: nn.Module, tracer: TraceContext) -> list[RemovableHandle]:
    handles = []
    
    target_types = (nn.Linear, nn.LayerNorm, nn.Conv2d, nn.Embedding, nn.GELU, nn.ReLU)
    
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue
            
        if not isinstance(module, target_types):
            continue
            
        # Hook for input
        def pre_hook(mod, args, mod_name=name):
            if not tracer.active:
                return
            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                with tracer.scope(mod_name):
                    tracer.tensor("input", args[0], operation=mod.__class__.__name__, module_path=mod_name, io="input")
                    
        # Hook for output
        def post_hook(mod, args, output, mod_name=name):
            if not tracer.active:
                return
            if isinstance(output, torch.Tensor):
                with tracer.scope(mod_name):
                    tracer.tensor("output", output, operation=mod.__class__.__name__, module_path=mod_name, io="output")
                    
        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))
        
    return handles

def remove_hooks(handles: list[RemovableHandle]) -> None:
    for h in handles:
        h.remove()
