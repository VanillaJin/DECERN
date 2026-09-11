import os

import torch.nn as nn
from timm.scheduler import create_scheduler
from torch.optim import Adam, SGD

from utils.checkpoint import save_json


def build_optimizer(args, exp_dir, model):
    optimizer_name = args.optimizer_name
    params = model.parameters()

    if optimizer_name == 'adam':
        opt_args = {
                'lr': args.lr,
                'weight_decay': args.weight_decay,
            }
        optimizer = Adam(params, **opt_args)

    elif optimizer_name == 'sgd':
        opt_args = {
                'lr': args.lr,
                'momentum': args.momentum,
                'weight_decay': args.weight_decay,
                'dampening': args.dampening,
                'nesterov': args.nesterov,
            }
        optimizer = SGD(params, **opt_args)

    else:
        raise NotImplementedError

    scheduler, _ = create_scheduler(args, optimizer)

    f_content = {'optimizer_name': optimizer_name, **opt_args}
    save_json(f_path=os.path.join(exp_dir, 'args', 'args_optimizer.json'), f_content=f_content)
    return optimizer, scheduler
