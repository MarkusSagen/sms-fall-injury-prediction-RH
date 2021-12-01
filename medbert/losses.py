#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss




class NLLabelSmoothing(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing=0.0):
        super(NLLabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing


    def forward(self, probs, target):
        logprobs = F.log_softmax(probs, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class LabelSmoothing(nn.Module):
    """Label Smooting Loss, based on Categorical CrossEntropy."""

    def __init__(self, classes, smoothing=0.0, dim=-1, weight=None):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim


    def forward(self, logit, target):
        assert 0 <= self.smoothing < 1
        pred = logit.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1.0))
            true_dist.scatter_(1, target.data.unsqueeze(1).type(torch.int64), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
