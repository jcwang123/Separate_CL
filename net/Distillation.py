import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T

from .feature_distill import *


# helper function
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def get_module_device(module):
    return next(module.parameters()).device


def default(val, def_val):
    return def_val if val is None else val


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


# mlp
class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_size),
                                 nn.BatchNorm1d(hidden_size),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_size, projection_size))

    def forward(self, x):
        return self.net(x)


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


# loss fn
def loss_kd(outputs, teacher_outputs, T=0.1):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
                             F.softmax(teacher_outputs / T, dim=1))
    return KD_loss


# main class
class MultiExpertsDistil(nn.Module):
    def __init__(self, dist_net, base_net, expert_nets, n_per_expert,
                 distill_type):
        super().__init__()
        self.chair_net = base_net
        self.dist_net = dist_net
        self.expert_nets = expert_nets
        self.distill_type = distill_type

        self.num_experts = len(self.expert_nets)

        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
            T.RandomGrayscale(p=0.2),
            RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2))

        self.augment1 = DEFAULT_AUG
        self.augment2 = DEFAULT_AUG

        self.n_per_expert = n_per_expert
        device = get_module_device(dist_net)
        self.to(device)

    def forward(self, x, class_id):
        image_one, image_two = self.augment1(x), self.augment2(x)
        b = image_one.size(0)
        loss = 0
        n_experts = b // self.n_per_expert
        for i in range(n_experts):
            for image in [image_one, image_two]:
                x = image[i * self.n_per_expert:(i + 1) * self.n_per_expert]
                k = class_id[i * self.n_per_expert]
                with torch.no_grad():
                    proj_chair, feature_chair = self.base_net.online_encoder(x)
                    proj_expert, feature_expert = self.expert_nets[
                        k].online_encoder(x)
                proj_dist, feature_dist = self.dist_net.online_encoder(x)

                if self.distill_type == 0:
                    pred_expert = self.expert_nets[k].online_predictor(
                        proj_dist)
                    loss_expert = loss_fn(pred_expert, proj_expert)
                    loss += loss_expert.mean()

                elif self.distill_type == 1:
                    loss_expert = loss_kd(feature_dist, feature_expert)
                    pred_chair = self.base_net.online_predictor(proj_dist)
                    loss_chair = loss_fn(pred_chair, proj_chair)
                    loss += (loss_expert + loss_chair)

                elif self.distill_type == 2:
                    pred_expert = self.expert_nets[k].online_predictor(
                        proj_dist)
                    pred_chair = self.base_net.online_predictor(proj_dist)
                    loss_chair = loss_fn(pred_chair, proj_chair)
                    loss_expert = loss_fn(pred_expert, proj_expert)
                    loss += (loss_expert + loss_chair).mean()
        loss /= n_experts * 2
        return loss
