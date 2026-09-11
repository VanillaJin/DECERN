import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import skew
from sklearn.cluster import KMeans
from torch.autograd import Variable

from strategies.base import BaseStrategy
from utils.checkpoint import save_npy
from utils.tools import time_usage_wrapper


class DECERNStrategy(BaseStrategy):
    def __init__(self, args, exp_dir, writer, **kwargs):
        super().__init__(args, exp_dir, writer, **kwargs)

        self.NC = self.data_info['n_classes']
        self.batch_size = self.data_info['batch_size']

        self.meth_anchor = args.meth_anchor
        self.meth_mask = args.meth_mask
        self.mask_topk = args.mask_topk
        self.lam_clamp = (1 / 64, 63 / 64)
        self.meth_uncertainty_score = args.meth_uncertainty_score
        self.meth_uncertainty_filter = args.meth_uncertainty_filter
        self.uncertainty_gamma = args.uncertainty_gamma
        self.meth_diversity_cluster = args.meth_diversity_cluster
        self.diversity_gamma = args.diversity_gamma

        self.save_vars = args.save_records

    def get_anchor(self, l_indices, model):
        r"""

        Args:
            l_indices (np.ndarray): The indices of labeled data.
            model (nn.Module): Task model.

        """
        l_ldr = self.data_adptr.get_loader_sample(l_indices)
        model.eval()

        l_prob = []
        l_repr = []
        with torch.no_grad():
            for batch in l_ldr:
                x = batch['x'].to(self.device)
                out, temp_repr, _ = model(x)

                l_prob.append(F.softmax(out, dim=1))
                l_repr.append(temp_repr)

        l_prob = torch.cat(l_prob, dim=0)
        l_repr = torch.cat(l_repr, dim=0)

        NC = self.NC
        NL, D = l_repr.size()

        a_prob = torch.zeros((NC, NC), device=l_prob.device)
        a_repr = torch.zeros((NC, D), device=l_repr.device)

        Y = torch.as_tensor(self.data_adptr.Y_tr[l_indices], device=l_prob.device)
        for i in range(NC):
            idx = Y == i
            count = idx.sum().item()
            a_prob[i] = l_prob[idx].mean(0) if count > 0 else l_prob.mean(0)
            a_repr[i] = l_repr[idx].mean(0) if count > 0 else l_repr.mean(0)

        return a_prob, a_repr

    def get_lam(self, u_prob):
        cap0, cap1 = self.lam_clamp
        lam = u_prob
        lam = torch.clamp(lam, cap0, cap1)
        return lam

    def get_dist(self, u_repr, a_repr):
        dist = (1 - (1 + F.cosine_similarity(u_repr.unsqueeze(1), a_repr.unsqueeze(0), dim=2)) / 2)
        return dist

    def get_mask(self, u_prob, u_repr, model, loss_fn):
        k = self.mask_topk
        D = u_repr.size(1)

        input_var = Variable(u_repr, requires_grad=True)
        target_var = Variable(torch.argmax(u_prob, dim=1), requires_grad=False)

        output, _, _ = model(input_var, is_repr=True)
        loss = (loss_fn(output, target_var)).mean(0)
        loss.backward(retain_graph=True)
        u_sign = input_var.grad

        u_sign = u_sign - torch.amin(u_sign, dim=1, keepdim=True)
        u_sign = F.normalize(u_sign, dim=1, p=2).unsqueeze(1)  # (B, 1, D)
        u_sign = F.adaptive_avg_pool1d(u_sign, 100)  # (B, 1, 100)
        _, indices = torch.topk(u_sign, k=k, dim=2)  # (B, 1, k)
        mask = torch.zeros_like(u_sign, dtype=torch.float32, device=self.device)  # (B, 1, 100)
        mask = torch.scatter(mask, 2, indices, 1.0)
        mask = F.interpolate(mask, D).squeeze(1)  # (B, D)

        ptn = torch.mean(mask, dim=1, keepdim=True)  # (B, 1)
        return mask, ptn

    def cpt_score(self, u_prob, b_prob, w_prob, m_prob, dist, ptn):
        m_prob_log = torch.log(m_prob + 1e-10)
        score_c = (torch.sum(-m_prob * m_prob_log, dim=1)) ** (1 - dist)
        mu = (torch.sum(-u_prob * m_prob_log, dim=1)) ** dist + score_c
        mb = (torch.sum(-b_prob * m_prob_log, dim=1)) ** dist + score_c
        mw = (torch.sum(-w_prob * m_prob_log, dim=1)) ** dist + score_c

        ptn = ptn.view(-1)
        score = ((mu * (1 - ptn) + mb * ptn) * 1 + mw * 1)
        return score

    def fusion_metric(self, cycle, budget, u_prob, a_prob, u_repr, a_repr, model):
        r"""Perform uncertainty sampling to select a subset of samples via local feature fusion.

        Args:
            cycle (int): Current cycle.
            budget (int): Sampling budget for active learning.
            u_prob (torch.Tensor): Unlabeled data prediction probability.
            a_prob (torch.Tensor): Anchor data prediction probability.
            u_repr (torch.Tensor): Unlabeled data feature representations.
            a_repr (torch.Tensor): Anchor data feature representations.
            model (nn.Module): Trained model.

        """
        model.eval()
        NC = self.NC
        NU = u_repr.size(0)
        B = self.batch_size
        gamma = self.uncertainty_gamma
        score = torch.zeros((NU, NC), device=self.device)
        loss_fn = nn.CrossEntropyLoss(reduction='none').to(self.device)

        for sta in range(0, NU, B):
            end = min(NU, sta + B)
            temp_len = end - sta

            u_repr_b = u_repr[sta:end, :]  # (B, D)
            u_prob_b = u_prob[sta:end, :]  # (B, NC)

            lam = self.get_lam(u_prob_b)  # (B, NC)
            dist = self.get_dist(u_repr_b, a_repr)  # (B, NC)
            mask, ptn = self.get_mask(u_prob_b, u_repr_b, model, loss_fn)  # (B, D), (B, 1)
            unchanged = (1 - mask) * u_repr_b  # (B, D)

            with torch.no_grad():
                for i in range(NC):
                    a_repr_b = a_repr[i, :].expand(temp_len, -1)  # (B, D)
                    a_prob_b = a_prob[i, :].expand(temp_len, -1)  # (B, NC)

                    lam_i = lam[:, i].view(-1, 1)  # (B, 1)
                    m_repr_b = unchanged + ((mask * u_repr_b) * (1 - lam_i) + (mask * a_repr_b) * lam_i)
                    out_mix, _, _ = model(m_repr_b, is_repr=True)
                    m_prob_b = F.softmax(out_mix, dim=1)
                    b_prob_b = (1 - lam_i) * u_prob_b + lam_i * a_prob_b
                    w_prob_b = (1 - ptn * lam_i) * u_prob_b + ptn * lam_i * a_prob_b
                    score[sta:end, i] = self.cpt_score(u_prob_b, b_prob_b, w_prob_b, m_prob_b, dist[:, i], ptn)

        score = torch.mean(score, dim=1).cpu().numpy()

        mean_ = np.mean(score)
        std_ = np.std(score)
        skew_ = skew(score)

        threshold = mean_ + std_ * skew_ * gamma[0]
        print(f'----- stats: mean : {mean_:.6f}\n'
              f'----- stats: std: {std_:.6f}\n'
              f'----- stats: skew: {skew_:.6f}\n'
              f'----- stats: threshold: {threshold:.6f}')

        idx = np.arange(NU)[score > threshold]
        if len(idx) < budget:
            idx = np.argsort(score)[::-1][:budget]
        sample_weight = score[idx.copy()]

        return idx, score, sample_weight

    def diversity_clustering(self, cycle, budget, u_repr, a_repr, sample_weight=None):
        r"""Perform diversity clustering to select a subset of samples based on their representativeness.

        Args:
            cycle (int): Current cycle.
            budget (int): Sampling budget for active learning.
            u_repr (torch.Tensor): Unlabeled data feature representations.
            a_repr (torch.Tensor): Anchor data feature representations.
            sample_weight (np.ndarray): The feature representations clustering weights of all candidate samples.

        """
        gamma = self.diversity_gamma
        B, NU = self.batch_size, u_repr.size(0)

        if self.meth_diversity_cluster == 'base':
            pass
        else:
            raise NotImplementedError

        # normalize
        u_repr = F.normalize(u_repr, p=2, dim=1)  # (NU, D)
        a_repr = F.normalize(a_repr, p=2, dim=1)  # (NC, D)

        # perform KMeans
        cluster = KMeans(n_clusters=budget)
        cluster.fit(u_repr.cpu().numpy(), sample_weight=sample_weight)
        cluster_idx = torch.tensor(cluster.labels_, device=self.device)  # (NU,)
        centers = F.normalize(torch.tensor(cluster.cluster_centers_, device=self.device), p=2, dim=1)  # (bdg, D)

        # compute the distance between each sample and anchor
        ua_score = torch.zeros(NU, device=self.device)
        for sta in range(0, NU, B):
            end = min(sta + B, NU)
            ua_score[sta:end] = 1.0 - torch.mm(u_repr[sta:end], a_repr.T).max(dim=1)[0]  # (B, NC) -> (B,)
        uc_score = torch.zeros_like(ua_score, device=self.device)

        cluster_masks = [torch.where(cluster_idx == i)[0] for i in range(budget)]
        selected_indices = []
        for i, cluster_mask in enumerate(cluster_masks):
            Ni = cluster_mask.size(0)
            if Ni == 0:
                continue
            elif Ni == 1:
                selected_indices.append(cluster_mask[0].item())
                continue

            # compute the distance between each sample and the center
            uc_score_i = 1.0 - torch.mv(u_repr[cluster_mask], centers[i])  # (NU_i,)
            ua_score_i = ua_score[cluster_mask]  # (NU_i,)
            uc_score[cluster_mask] = uc_score_i

            uc_score_i = (uc_score_i - uc_score_i.min()) / (uc_score_i.max() - uc_score_i.min() + 1e-10)
            ua_score_i = (ua_score_i - ua_score_i.min()) / (ua_score_i.max() - ua_score_i.min() + 1e-10)

            # combine score
            comb_score = gamma[0] * uc_score_i - (1 - gamma[0]) * ua_score_i  # (NU_i,)

            # select
            selected_indices.append(cluster_mask[torch.argmin(comb_score, dim=0)].item())

        return np.array(selected_indices)

    @time_usage_wrapper
    def sample(self, cycle, budget, l_indices, u_indices):
        model = self.model

        u_prob, u_repr = self._get_prob_repr(u_indices, model)
        u_prob, u_repr = u_prob.to(self.device), u_repr.to(self.device)
        a_prob, a_repr = self.get_anchor(l_indices, model)
        torch.cuda.empty_cache()

        idx1, _, sample_weight = self.fusion_metric(cycle, budget, u_prob, a_prob, u_repr, a_repr, model)
        print(f"-------------------- Uncertainty sampling: {len(idx1)} / {len(u_indices)}")
        del a_prob
        torch.cuda.empty_cache()

        if len(idx1) > budget:
            idx2 = self.diversity_clustering(cycle, budget, u_repr[idx1.copy()], a_repr, sample_weight)
            select_idx = idx1[idx2]
            print(f"-------------------- Diversity sampling: {len(idx2)} / {len(idx1)}")
        else:
            select_idx = idx1
        s_indices = u_indices[select_idx]

        if len(s_indices) < budget:
            choice_size = budget - len(s_indices)
            choice_a = np.setdiff1d(u_indices, s_indices)
            plus_idx = np.random.choice(choice_a, choice_size, replace=False)
            s_indices = np.concatenate((s_indices, plus_idx), axis=0)
            print(f"-------------------- Random sampling: {choice_size} / {len(choice_a)}")

        return s_indices, select_idx, u_repr.cpu(), u_prob.cpu()
