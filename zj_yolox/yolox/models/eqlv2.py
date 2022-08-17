import torch
import torch.nn as nn
import torch.nn.functional as F


class EQLv2(nn.Module):
    def __init__(self, num_classes=5):
        super(EQLv2, self).__init__()
        self.gamma = 12
        self.mu = 0.8
        self.alpha = 2.0
        self._pos_grad = None
        self._neg_grad = None
        self.pos_neg = None
        self.num_classes = num_classes
        self.last_pos_neg = 1
        self.pre_pos_neg = 1

    def map_func(self, x):
        return 1 / (1 + torch.exp(-self.gamma * (x - self.mu)))

    # calculate g(t)
    def collect_grad(self, pred, target, weight):
        prob = torch.sigmoid(pred)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)

        # do not collect grad for objectiveness branch [:-1]
        pos_grad = torch.sum(grad * target * weight, dim=0, keepdim=False)  # [:-1]
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0, keepdim=False)  # [:-1]

        # dist.all_reduce(pos_grad)
        # dist.all_reduce(neg_grad)
        self._pos_grad += pos_grad
        self._neg_grad += neg_grad
        self.last_pos_neg = self.pre_pos_neg
        self.pre_pos_neg = self.pos_neg
        self.pos_neg = self._pos_grad / (self._neg_grad + 1e-10)

    def get_weight(self, pred):
        # we do not have information about pos grad and neg grad at beginning
        if self._pos_grad is None:
            self._pos_grad = pred.new_zeros(self.num_classes)
            self._neg_grad = pred.new_zeros(self.num_classes)
            neg_w = pred.new_ones((self.n_i, self.n_c))
            pos_w = pred.new_ones((self.n_i, self.n_c))
        else:
            # the negative weight for objectiveness is always 1
            neg_w = self.map_func(self.pos_neg)  # torch.cat([self.map_func(self.pos_neg), pred.new_ones(1)])
            pos_w = 1 + self.alpha * (1 - neg_w)
            neg_w = neg_w.reshape(1, -1).expand(self.n_i, self.n_c)
            pos_w = pos_w.reshape(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w

    def forward(self, pred, true):
        self.n_i, self.n_c = pred.size()
        self.gt_classes = true
        self.pred_class_logits = pred

        target = self.gt_classes
        pos_w, neg_w = self.get_weight(pred)

        weight = pos_w * target + neg_w * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(pred, target,reduction='none')

        cls_loss = torch.sum(cls_loss * weight) / self.n_i

        self.collect_grad(pred.detach(), target.detach(), weight.detach())

        return cls_loss


if __name__ == '__main__':
    # target = pred.new_zeros(self.n_i, self.n_c)
    # target[torch.arange(self.n_i), gt_classes.int().detach().cpu()] = 1
    # return target
    print(1)