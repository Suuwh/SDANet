#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/clovaai/voxceleb_trainer/tree/master/loss

import numpy as np, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# DLBLoss
class DLBLoss(nn.Module):
    # nOut = 512, nClasses = 9
    def __init__(self, nOut, nClasses, **kwargs):
        super(DLBLoss, self).__init__()
        # self.model = model
        # self.test_normalize = True

        self.criterion = torch.nn.CrossEntropyLoss()
        self.fc = nn.Linear(nOut, nClasses, bias=True)

        self.last_images = None
        self.last_logits = None
        self.last_lable = None
        self.T = 0.0

    # print('Initialised Softmax Loss')

    def forward(self, x, label=None, images=None, epoch=0, last=False):
        target = label
        # i = images.shape[0]
        # x = self.fc(x)

        if not last:
            # self.last_images = None
            # self.last_logits = None
            # self.last_lable = None

            batch_size = images.shape[0]
            logit = self.fc(x)
            loss_org = self.criterion(logit, target)
            loss_dlb = torch.tensor(0.0)
        else:
            # images = torch.cat([images, self.last_images], dim=0)
            batch_size = int(images.shape[0] / 2)
            logit = self.fc(x)

            logit, logit_last = logit[:batch_size], logit[batch_size:]
            loss_org = self.criterion(logit, target)

            # self.T = 3 - ( epoch / 40 )
            self.T = 3
            #self.T = 3 + (epoch / 40)  # 3 ~ 5

            loss_dlb = (
                    F.kl_div(
                        F.log_softmax(logit_last / self.T, dim=1),
                        self.last_logits,
                        reduction="batchmean",
                    )
                    * self.T
                    * self.T
            )

        # Update last
        self.last_images = images[:batch_size].detach()
        self.last_logits = torch.softmax(logit.detach() / self.T, dim=1)
        self.last_lable = target

        a = 2 * ( epoch / 80 )
        loss = loss_org + a * loss_dlb
        # nloss = self.criterion(x, label)
        return loss


class Softmax(nn.Module):
    def __init__(self, nOut, nClasses, **kwargs):
        super(Softmax, self).__init__()

        self.test_normalize = True

        self.criterion = torch.nn.CrossEntropyLoss()
        self.fc = nn.Linear(nOut, nClasses, bias=False)

        print('Initialised Softmax Loss')

    def forward(self, x, label=None):
        x = self.fc(x)
        nloss = self.criterion(x, label)
        return nloss


class AMSoftmax(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.3, scale=15, **kwargs):
        super(AMSoftmax, self).__init__()

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.W = torch.nn.Parameter(torch.randn(nOut, nClasses), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised AMSoftmax m=%.3f s=%.3f' % (self.m, self.s))

    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, label)

        return loss


class AAMSoftmax(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.3, scale=15, easy_margin=False, **kwargs):
        super(AAMSoftmax, self).__init__()

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        print('Initialised AAMSoftmax margin %.3f scale %.3f' % (self.m, self.s))

    def forward(self, x, label=None):
        code_dim = x.size()[-1]
        x = x.reshape(-1, code_dim)

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        # one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)

        return loss


class CircleLoss(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.2, scale=30, **kwargs):
        super(CircleLoss, self).__init__()
        self.margin = margin
        self.gamma = scale
        self.soft_plus = nn.Softplus()
        self.class_num = nClasses
        self.emdsize = nOut

        self.weight = nn.Parameter(torch.FloatTensor(self.class_num, self.emdsize))
        nn.init.xavier_uniform_(self.weight)
        self.use_cuda = True

    def forward(self, x, label=None):
        similarity_matrix = nn.functional.linear(nn.functional.normalize(x, p=2, dim=1, eps=1e-12),
                                                 nn.functional.normalize(self.weight, p=2, dim=1, eps=1e-12))

        if self.use_cuda:
            one_hot = torch.zeros(similarity_matrix.size(), device='cuda')
        else:
            one_hot = torch.zeros(similarity_matrix.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot = one_hot.type(dtype=torch.bool)
        # sp = torch.gather(similarity_matrix, dim=1, index=label.unsqueeze(1))
        sp = similarity_matrix[one_hot]
        mask = one_hot.logical_not()
        sn = similarity_matrix[mask]

        sp = sp.view(x.size()[0], -1)
        sn = sn.view(x.size()[0], -1)

        ap = torch.clamp_min(-sp.detach() + 1 + self.margin, min=0.)
        an = torch.clamp_min(sn.detach() + self.margin, min=0.)

        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1))

        return loss.mean()
