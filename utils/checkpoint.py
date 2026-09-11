import json
import os
from typing import Union

import numpy as np
import torch
import torch.nn as nn


def save_ckpt(args, exp_dir, cycle, tag_scalar_dict, model):
    f_path = os.path.join(exp_dir, 'checkpoints', f'ckpt_cyc{cycle}.pth')

    obj = {
        'cycle': cycle,
        'metric': tag_scalar_dict,
        'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
    }
    torch.save(obj, f_path)


def save_json(f_path: str, f_content: dict):
    with open(f_path, 'w') as f:
        json.dump(f_content, f, indent=4)


def save_npy(f_path: str, f_content: Union[torch.Tensor, np.ndarray, list]):
    if isinstance(f_content, torch.Tensor):
        f_content = f_content.detach().cpu().numpy()
    elif isinstance(f_content, np.ndarray):
        pass
    elif isinstance(f_content, list):
        f_content = np.array(f_content)
    else:
        raise NotImplementedError

    np.save(f_path, f_content)
