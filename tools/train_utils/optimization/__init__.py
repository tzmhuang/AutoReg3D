from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle, CosineAnnealing, ConstantLR

def get_parameter_names(model, forbidden_layer_types, forbidden_layer_names=None):
    """
    # from hf transformers
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    if forbidden_layer_names is None:
        forbidden_layer_names = []
    result = []
    for name, child in model.named_children():
        child_params = get_parameter_names(child, forbidden_layer_types, forbidden_layer_names)
        result += [
            f"{name}.{n}"
            for n in child_params
            if not isinstance(child, tuple(forbidden_layer_types))
            and not any(forbidden in f"{name}.{n}".lower() for forbidden in forbidden_layer_names)
        ]
    # Add model specific parameters that are not in any child
    result += [
        k for k in model._parameters.keys() if not any(forbidden in k.lower() for forbidden in forbidden_layer_names)
    ]
    return result

def get_decay_parameter_names(model):
    """
    Get all parameter names that weight decay will be applied to

    Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
    apply to those modules since this function only filter out instance of nn.LayerNorm
    """
    ALL_FORBID_LAYERS = (nn.LayerNorm, nn.Embedding, nn.BatchNorm1d, 
                                    nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    decay_parameters = get_parameter_names(model, ALL_FORBID_LAYERS)
    # exclude bias parameters
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    # exclude embedding parameters
    decay_parameters = [name for name in decay_parameters if "embed" not in name]
    decay_parameters = [name for name in decay_parameters if "wpe" not in name] # pos embed
    decay_parameters = [name for name in decay_parameters if "wte" not in name] # token embed
    decay_parameters = [name for name in decay_parameters if "query_tokens" not in name] # qformer query tokens
    return decay_parameters


def build_optimizer(model, optim_cfg):
    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'adamW':
        betas = tuple(optim_cfg.get('BETAS', (0.9, 0.99)))
        decay_names = set(get_decay_parameter_names(model))
        decay_params    = [p for n, p in model.named_parameters() if p.requires_grad and n in decay_names]
        no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and n not in decay_names]
        param_groups = [
            {'params': decay_params,    'weight_decay': optim_cfg.WEIGHT_DECAY},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        # Raw optimizer (no OptimWrapper): adamW pairs with LambdaLR, which drives
        # param_groups directly. AdamW applies decoupled weight decay per group.
        optimizer = optim.AdamW(param_groups, lr=optim_cfg.LR, betas=betas)
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    elif optim_cfg.OPTIMIZER == 'adamW_onecycle':
        betas = tuple(optim_cfg.get('BETAS', (0.9, 0.99)))
        decay_names = set(get_decay_parameter_names(model))
        decay_params    = [p for n, p in model.named_parameters() if p.requires_grad and n in decay_names]
        no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and n not in decay_names]
        param_groups = [
            {'params': decay_params,    'weight_decay': optim_cfg.WEIGHT_DECAY},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        opt = optim.AdamW(param_groups, lr=optim_cfg.LR, betas=betas)
        # Wrapper exists only to expose .lr/.mom setters for OneCycle; AdamW handles
        # decoupled weight decay natively per group. bn_wd=False prevents the wd
        # setter from broadcasting WD onto the no-decay group at init.
        optimizer = OptimWrapper(opt, wd=optim_cfg.WEIGHT_DECAY, true_wd=False, bn_wd=False)

    elif optim_cfg.OPTIMIZER in ['adam_onecycle','adam_cosineanneal']:
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]
        betas = optim_cfg.get('BETAS', (0.9, 0.99))
        betas = tuple(betas)
        optimizer_func = partial(optim.Adam, betas=betas)
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    if optim_cfg.OPTIMIZER == 'adamW':
        lr_scheduler = ConstantLR(optimizer)
    elif optim_cfg.OPTIMIZER in ['adam_onecycle', 'adamW_onecycle']:
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    elif optim_cfg.OPTIMIZER == 'adam_cosineanneal':
        lr_scheduler = CosineAnnealing(
            optimizer, total_steps, total_epochs, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.PCT_START, optim_cfg.WARMUP_ITER
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler
