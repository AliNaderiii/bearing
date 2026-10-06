import os
import sys
import json
import datetime
import torch
import torch.optim as optim
from typing import Optional, List, Dict

# --- Helper to find parameter grouping function ---
def get_group_parameters_func():
    """
    Tries to import group_parameters_dict from utils.train_function.
    Falls back to a default grouping strategy if not found.
    """
    try:
        from utils.train_function import group_parameters_dict
        return group_parameters_dict
    except ImportError:
        try:
            # Try importing from local directory if utils package fails
            from train_function import group_parameters_dict
            return group_parameters_dict
        except ImportError:
            print("[Warning] 'group_parameters_dict' not found in utils.train_function.")
            print("Using default parameter grouping (Bias/BN weight decay = 0).")
            return default_group_parameters

def default_group_parameters(model: torch.nn.Module) -> Dict[str, List[torch.nn.Parameter]]:
    """
    Standard grouping: Separate parameters into those with weight decay and those without (bias, bn).
    """
    decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Check for bias or BatchNorm parameters (usually no weight decay)
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
            
    return {
        'decay': decay,
        'no_decay': no_decay
    }

# --- Main Optimizer Function ---
def get_optimizer(model: torch.nn.Module, config, optimizer_name: str = 'Adam', param_groups: Optional[list] = None):
    # Config setup with defaults
    base_lr = getattr(config, 'lr', 1e-3)
    alpha = getattr(config, 'alpha', 0.09) # Specific multiplier if needed
    weight_decay = getattr(config, 'weight_decay', 4e-6)

    if param_groups is None:
        group_func = get_group_parameters_func()
        groups = group_func(model)
        
        param_groups = []
        
        # Strategy 1: If using the complex grouping from your original code
        if 'weight_r' in groups: 
            # Add groups based on specific keys (Legacy support)
            def add_legacy(name, params, lr_mult, wd):
                if params:
                    param_groups.append({'params': params, 'lr': base_lr * lr_mult, 'weight_decay': wd, 'name': name})

            add_legacy('weight_r', groups.get('weight_r', []), 1.0, weight_decay)
            add_legacy('weight_g', groups.get('weight_g', []), alpha, weight_decay)
            add_legacy('weight_b', groups.get('weight_b', []), alpha, weight_decay)
            
            # Biases & BN (usually 0 weight decay)
            add_legacy('bias_r', groups.get('bias_r', []), 1.0, 0.0)
            add_legacy('bias_g', groups.get('bias_g', []), alpha, 0.0)
            add_legacy('bias_b', groups.get('bias_b', []), alpha, 0.0)
            add_legacy('bn', groups.get('bn', []), 1.0, 0.0)
            
            # FC layers
            add_legacy('fc_weights', groups.get('fc_weights', []), 1.0, weight_decay)
            add_legacy('fc_biases', groups.get('fc_biases', []), 1.0, 0.0)
            
            # Catch-all
            add_legacy('others', groups.get('others', []), 1.0, weight_decay)

        # Strategy 2: Default simple grouping (Decay vs No Decay)
        elif 'decay' in groups:
             if groups['decay']:
                 param_groups.append({'params': groups['decay'], 'weight_decay': weight_decay, 'lr': base_lr})
             if groups['no_decay']:
                 param_groups.append({'params': groups['no_decay'], 'weight_decay': 0.0, 'lr': base_lr})
        
        # Fallback: Just all parameters
        if not param_groups:
            param_groups = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': base_lr, 'weight_decay': weight_decay}]

    # Logging
    try:
        log_data = {
            'optimizer': optimizer_name,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'groups': []
        }
        for i, g in enumerate(param_groups):
            # Count total number of elements (weights) in the group
            n_params = sum(p.numel() for p in g['params'])
            log_data['groups'].append({
                'id': i,
                'name': g.get('name', 'unnamed'),
                'n_params': n_params,
                'lr': g.get('lr', base_lr),
                'weight_decay': g.get('weight_decay', weight_decay)
            })
        
        # Uncomment to save log file
        # with open(f"optimizer_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        #     json.dump(log_data, f, indent=2)
            
        print(f"[Optimizer] Selected: {optimizer_name} with {len(param_groups)} parameter groups.")
    except Exception as e:
        print(f"[Optimizer] Logging failed: {e}")

    # Create Optimizer
    if optimizer_name == 'SGD':
        return optim.SGD(param_groups, momentum=0.9)
    elif optimizer_name == 'Adam':
        return optim.Adam(param_groups, lr=base_lr)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(param_groups, lr=base_lr)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(param_groups, lr=base_lr, alpha=0.9)
    elif optimizer_name == 'AMSGradP':
        try:
            from AdamP_amsgrad import AdamP
            return AdamP(param_groups, lr=base_lr, weight_decay=weight_decay)
        except ImportError:
            print("[Error] AdamP_amsgrad not found. Falling back to AdamW.")
            return optim.AdamW(param_groups, lr=base_lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

# --- Self Test ---
if __name__ == '__main__':
    # Simple dummy model for testing
    model = torch.nn.Sequential(
        torch.nn.Conv1d(1, 10, 3),
        torch.nn.BatchNorm1d(10),
        torch.nn.Linear(10, 2)
    )
    
    class Config:
        lr = 0.01
        weight_decay = 1e-4
        alpha = 0.5
        
    opt = get_optimizer(model, Config(), optimizer_name='AdamW')
    print("Optimizer created successfully.")
    print(opt)
