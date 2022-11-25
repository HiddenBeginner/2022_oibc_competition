import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def _unroll_batch(self, input, target):
        input = input.reshape((-1, 2))
        target = target.reshape(-1)

        return input, target

    def forward(self, input, target):
        raise NotImplementedError


class CompetitionMetric(BaseLoss):
    def __init__(self):
        super(CompetitionMetric, self).__init__()

    def loss1(self, input, target):
        lower, upper = input[:, 0], input[:, 1]

        return torch.abs((lower + upper) / 2.0 - target)

    def loss2(self, input):
        lower, upper = input[:, 0], input[:, 1]

        return upper - lower

    def loss3(self, input, target):
        lower, upper = input[:, 0], input[:, 1]

        return target * (lower > target) + target * (upper < target)

    @torch.no_grad()
    def forward(self, input, target):
        input, target = self._unroll_batch(input, target)
        loss1 = self.loss1(input, target)
        loss2 = self.loss2(input)
        loss3 = self.loss3(input, target)

        return {
            'TotalMetric': (0.45 * loss1 + 0.45 * loss2 + 0.1 * loss1).sum(),
            'Metric1': loss1.sum(),
            'Metric2': loss2.sum(),
            'Metric3': loss3.sum()
        }


class MeanStdLoss(BaseLoss):
    def __init__(self, k=1.0, weights=(1.0, 0.2, 0.7), **kwargs):
        super(MeanStdLoss, self).__init__()
        self.k = k
        self.weights = weights

    def loss1(self, input, target):
        return torch.abs(input[:, 0] - target)

    def loss2(self, input):
        return 2 * self.k * input[:, 1]

    def loss3(self, input, target):
        return (
            F.relu(target - (input[:, 0] + self.k * input[:, 1])) +
            F.relu((input[:, 0] - self.k * input[:, 1]) - target)
        )

    def forward(self, input, target):
        input, target = self._unroll_batch(input, target)
        loss1 = self.loss1(input, target)
        loss2 = self.loss2(input)
        loss3 = self.loss3(input, target)

        return (self.weights[0] * loss1 + self.weights[1] * loss2 + self.weights[2] * loss3).sum()


class NLGLLoss(BaseLoss):
    """Negative Log Gaussian Likelihood Loss"""
    def __init__(self, **kwargs):
        super(NLGLLoss, self).__init__()

    def forward(self, input, target):
        input, target = self._unroll_batch(input, target)
        mu = input[:, 0]
        std = input[:, 1] + 1e-6
        m = Normal(mu, std)

        return -m.log_prob(target).mean()
