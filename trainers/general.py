import copy
import os

import numpy as np
import torch
import torch.nn as nn
from timm.utils import AverageMeter, accuracy

from trainers.base import BaseTrainer
from utils import build_model, build_optimizer
from utils.checkpoint import save_npy


class GeneralClassifierTrainer(BaseTrainer):
    def __init__(self, args, exp_dir, writer, **kwargs):
        super().__init__(args, exp_dir, writer, **kwargs)

    def _build_net(self):
        self.model = build_model(self.args, self.exp_dir, self.data_info)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer, self.scheduler = build_optimizer(self.args, self.exp_dir, self.model)

        if self._reset == 'p_cycle_same_init':
            if self.model_params is None:
                self.model_params = copy.deepcopy(self.model.state_dict())
            else:
                self.model.load_state_dict(copy.deepcopy(self.model_params))
        else:
            raise NotImplementedError

        self.model = self.model.to(self.device)

    def _train_epoch(self, epoch, ldr_l, ldr_u=None):
        self.model.train()

        acc1_meter = AverageMeter()
        loss_meter = AverageMeter()

        for batch in ldr_l:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)

            out, _, _ = self.model(x)

            self.optimizer.zero_grad()
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            acc1 = accuracy(out, y, topk=(1,))[0]
            acc1_meter.update(acc1.item(), y.size(0))
            loss_meter.update(loss.item(), y.size(0))

        if self.scheduler is not None:
            self.scheduler.step(epoch)

        return {
            'acc1': acc1_meter.avg,
            'loss': loss_meter.avg,
        }

    @torch.no_grad()
    def _test_epoch(self, cycle, ldr_te):
        self.model.eval()

        acc1_meter = AverageMeter()
        # acc3_meter = AverageMeter()
        # acc5_meter = AverageMeter()
        #
        # y_true = []
        # y_prob = []
        # y_repr = []

        # from tqdm import tqdm
        # for batch in tqdm(ldr_te):
        for batch in ldr_te:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)

            temp_out, temp_repr, _ = self.model(x)

            acc1, acc3, acc5 = accuracy(temp_out, y, topk=(1, 3, 5))
            acc1_meter.update(acc1.item(), y.size(0))
            # acc3_meter.update(acc3.item(), y.size(0))
            # acc5_meter.update(acc5.item(), y.size(0))
            #
            # y_true.append(y.cpu().numpy())
            # y_prob.append(temp_out.softmax(dim=1).cpu().numpy())
            # y_repr.append(temp_repr.cpu().numpy())

        return {
            "acc1": acc1_meter.avg,
            # "acc3": acc3_meter.avg,
            # "acc5": acc5_meter.avg,
        }

