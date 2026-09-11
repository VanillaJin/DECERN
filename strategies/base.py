import os
from abc import abstractmethod
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.tools import DEVICE


class BaseStrategy:
    def __init__(self, args, exp_dir, writer, data_info, data_adptr):
        r"""

        Args:
            args (argparse.Namespace): Command line arguments.
            exp_dir (str): Directory to save checkpoints and logs.
            writer (SummaryWriter): Tensorboard summary writer.
            data_info (dict): Data Meta Information Dictionary.
            data_adptr (utils.adpt_data.DataLoaderAdapter): .

        """
        self.args = args
        self.exp_dir = exp_dir
        self.writer = writer
        self.device = DEVICE

        self.data_info = data_info
        self.data_adptr = data_adptr
        self.model = None

        self.meth_sample0 = args.sample0

        print(f'Info | Output path: {exp_dir}')

    @torch.no_grad()
    def _get_repr(self, indices, model):
        r"""

        Args:
            indices (np.ndarray): The index of data.
            model (nn.Module): Trained model.

        """
        ldr = self.data_adptr.get_loader_sample(indices)
        model.eval()

        repr_ = []
        for batch in ldr:
            x = batch['x'].to(self.device)
            _, temp_repr_, _ = model(x)

            repr_.append(temp_repr_.detach().cpu())
        repr_ = torch.cat(repr_, dim=0)

        return repr_

    @torch.no_grad()
    def _get_prob_repr(self, indices, model):
        r"""

        Args:
            indices (np.ndarray): The index of data.
            model (nn.Module): Trained model.

        """
        ldr = self.data_adptr.get_loader_sample(indices)
        model.eval()

        prob_ = []
        repr_ = []

        for batch in ldr:
            x = batch['x'].to(self.device)
            out, temp_repr_, _ = model(x)
            temp_prob_ = F.softmax(out, dim=1)

            prob_.append(temp_prob_.detach().cpu())
            repr_.append(temp_repr_.detach().cpu())

        prob_ = torch.cat(prob_, dim=0)
        repr_ = torch.cat(repr_, dim=0)
        return prob_, repr_

    def sample0(self, t_indices):
        r"""Active Learning Sampling at cycle0

        Args:
            t_indices (np.ndarray): The index of the training data.

        """
        mode = self.meth_sample0
        available_N = len(t_indices)
        NC = self.data_info['n_classes']
        budget = self.data_info['n_init_bdg']
        indices_dir = os.path.join(self.exp_dir, 'indices')

        if 0 < budget < available_N:
            if mode == 'r':
                temp_idx = np.arange(len(t_indices))
                np.random.shuffle(temp_idx)
                l_indices = t_indices[temp_idx[:budget]]
                u_indices = t_indices[temp_idx[budget:]]

            else:
                raise NotImplementedError

        elif available_N <= budget:
            print(f'Info | Sampling0 | l_indices size is {available_N}, u_indices size is 0.')
            l_indices = t_indices
            u_indices = np.array([], dtype=int)

        elif budget == 0:
            print(f'Info | Sampling0 | l_indices size is 0, u_indices size is {available_N}.')
            l_indices = np.array([], dtype=int)
            u_indices = t_indices

        else:
            raise NotImplementedError

        if len(l_indices) > 0:
            np.save(os.path.join(indices_dir, f'ind_cyc0_L{len(l_indices)}.npy'), l_indices)
        # if len(u_indices) > 0:
        #     np.save(os.path.join(indices_dir, f'ind_cyc0_U{len(u_indices)}.npy'), u_indices)
        return l_indices, u_indices

    def pre_sample(self, cycle, u_indices):
        r"""Active Learning Pre-sampling.

        Args:
            cycle (int): Current active learning cycle.
            u_indices (np.ndarray): The index of the unlabeled data.

        """
        available_N = len(u_indices)
        pre_sample_N = self.data_info['pre_sample']
        indices_dir = os.path.join(self.exp_dir, 'indices')

        if (available_N <= pre_sample_N) or (pre_sample_N < 0):
            p_indices = u_indices

        elif 0 < pre_sample_N < available_N:
            print(f'Info | Pre-sampling{cycle} | p_indices size is {pre_sample_N}, u_indices size is {available_N}.')
            temp_idx = np.arange(len(u_indices))
            np.random.shuffle(temp_idx)
            p_indices = u_indices[temp_idx[:pre_sample_N]]

        else:
            raise NotImplementedError

        if len(p_indices) > 0:
            np.save(os.path.join(indices_dir, f'ind_cyc{cycle}_P{len(p_indices)}.npy'), p_indices)
        return p_indices

    @abstractmethod
    def sample(self, cycle, budget, l_indices, u_indices):
        r"""Active Learning Sampling.

        Perform the active learning strategy to obtain a subset of unlabelled samples.

        Args:
            cycle (int): Current active learning cycle.
            budget (int): Sampling budget for active learning.
            l_indices (np.ndarray): The index of the labeled data.
            u_indices (np.ndarray): The index of the unlabeled data.

        """
        pass

    def update(self, cycle, l_indices, u_indices, s_indices):
        r"""Labelling and updating.

        Labelling the unlabelled samples selected by the active learning strategy and
        updating the labelled and unlabelled data.

        Args:
            cycle (int): Current active learning cycle.
            l_indices (np.ndarray): The index of the labeled data.
            u_indices (np.ndarray): The index of the unlabeled data.
            s_indices (np.ndarray): The index of the sampled data.

        """
        indices_dir = os.path.join(self.exp_dir, 'indices')

        cur_s_indices = s_indices
        new_l_indices = np.append(l_indices, s_indices)
        new_u_indices = np.setdiff1d(u_indices, s_indices)

        np.save(os.path.join(indices_dir, f'ind_cyc{cycle}_S{len(cur_s_indices)}.npy'), cur_s_indices)
        np.save(os.path.join(indices_dir, f'ind_cyc{cycle}_L{len(new_l_indices)}.npy'), new_l_indices)
        # np.save(os.path.join(indices_dir, f'ind_cyc{cycle}_U{len(new_u_indices)}.npy'), new_u_indices)
        assert (len(new_l_indices) - len(l_indices)) == (len(u_indices) - len(new_u_indices)), \
            (f'l_indices add ({len(new_l_indices)} - {len(l_indices)} = {len(new_l_indices) - len(l_indices)}), \n'
             f'u_indices sub ({len(u_indices)} - {len(new_u_indices)} = {len(u_indices) - len(new_u_indices)})')
        return new_l_indices, new_u_indices

    def syn_model(self, model):
        r"""Model synchronisation.

        Receive trained models and modules from trainer.

        Args:
            model (Tuple[nn.Module]): Trained model.

        """
        self.model = model
