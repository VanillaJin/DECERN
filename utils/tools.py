import os
import random
import time
from functools import wraps

import numpy as np
import torch
import torch.backends.cudnn


def set_env(args, seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


def make_exp_dir(args, seed=1):
    exp_name = ''

    # data + arch + active learning strategy
    exp_name += f'{args.data_name.replace("_", "")}'
    if args.model_name.startswith('ViT'):
        exp_name += f'_ViT'
    elif args.model_name.startswith('ResNet'):
        exp_name += f'_RN'
    else:
        exp_name += f'_{args.model_name}'
    exp_name += f'_{args.strategy_name}'

    # TRAINER
    exp_name += f'_seed{seed}'

    # ACTIVE LEARNING STRATEGY
    if (0 < args.init_budget_x) and (args.init_budget_x == args.comm_budget_x):
        exp_name += f'_bdg{args.init_budget_x}x'

    if args.remarks != '.':
        exp_name += f'_remarks_{args.remarks}'

    exp_dir = os.path.join(args.logs_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'args'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'indices'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'records'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'tensorboard'), exist_ok=True)
    return exp_dir


def get_init_TV_indices(exp_dir, data_info):
    n_data = data_info['n_data']
    n_val = data_info['n_valid']
    tol_indices = list(range(n_data))

    if n_val > 0:
        v_inx = -n_val
        t_indices = np.array(tol_indices[:v_inx])
        v_indices = np.array(tol_indices[v_inx:])
        np.random.shuffle(t_indices)
    else:
        random.shuffle(tol_indices)
        t_indices = np.array(tol_indices[:])
        v_indices = None

    np.save(os.path.join(exp_dir, 'indices', f'ind_cyc0_T{len(t_indices)}.npy'), t_indices)
    if v_indices is not None:
        np.save(os.path.join(exp_dir, 'indices', f'ind_cyc0_V{len(v_indices)}.npy'), v_indices)
    return t_indices, v_indices


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def time_usage_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        sta_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_used = end_time - sta_time

        print(f"----- Monitor | time using | {func.__name__}: {time_used:.6f} s")
        return result
    return wrapper
