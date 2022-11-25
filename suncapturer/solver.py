import math
import os
from glob import glob

import torch
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import _LRScheduler

from .losses import CompetitionMetric, MeanStdLoss, NLGLLoss

loss_dict = {
    'MeanStdLoss': MeanStdLoss,
    'NLGLLoss': NLGLLoss
}


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.k = config['k']
        self.epochs = config['epochs']
        self.dir_ckpt = config['dir_ckpt']

        self.optimizer = optim.Adam(self.model.parameters(), **config['optimizer'])
        self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer, **config['scheduler'])

        self.criterion = loss_dict[config['loss']['loss']](**config['loss'])
        self.metric = CompetitionMetric()

        self.best_metric = 9999.
        wandb.init(**config['wandb'], config=config)

    def fit(self, train_loader, valid_loader, test_loader=None):
        for e in range(self.epochs):
            self.train_one_epoch(train_loader)
            train_metric = self.evaluate(train_loader)
            valid_metric = self.evaluate(valid_loader)

            log = {'Epochs': e + 1}
            for k, v in train_metric.items():
                log[f'train_{k}'] = v

            for k, v in valid_metric.items():
                log[f'valid_{k}'] = v

            if test_loader is not None:
                test_metric = self.evaluate(test_loader)
                for k, v in test_metric.items():
                    log[f'test_{k}'] = v

            log['LR'] = self.scheduler.get_lr()[0]
            wandb.log(log)

            self.save(f'{self.dir_ckpt}/last_ckpt.bin')
            if valid_metric['TotalMetric'] < self.best_metric:
                self.best_metric = valid_metric['TotalMetric']
                self.save(f'{self.dir_ckpt}/best_ckpt_{e}.bin')
                # Keep top 3 models
                for path in sorted(glob(f'{self.dir_ckpt}/best_ckpt_*.bin'))[:-3]:
                    os.remove(path)

            self.scheduler.step()

        wandb.finish()

    def train_one_epoch(self, loader):
        self.model.train()
        for X, y in loader:
            out = self.model(X)
            loss = self.criterion(out, y)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        n = 0
        scores = {'TotalMetric': 0, 'Metric1': 0, 'Metric2': 0, 'Metric3': 0, 'Loss': 0}
        self.model.eval()
        for X, y in loader:
            out = self.model(X)
            bound = torch.zeros_like(out)
            bound[:, :, 0] = out[:, :, 0] - self.k * out[:, :, 1]
            bound[:, :, 1] = out[:, :, 0] + self.k * out[:, :, 1]

            loss = self.criterion(out, y)
            score = self.metric(bound, y)

            n += len(y)
            scores['Loss'] += loss
            for k, v in score.items():
                scores[k] += v

        for k, v in scores.items():
            scores[k] = v / n

        return scores

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    """
    https://gaussian37.github.io/dl-pytorch-lr_scheduler/
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * 
                    (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
