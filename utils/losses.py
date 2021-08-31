import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_cross_entropyloss(prob, target):
    loss = -(target * torch.log(prob) + (1 - target) * (torch.log(1 - prob)))
    loss = torch.sum(loss) / torch.numel(target)
    return loss


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        predict = torch.clamp(predict, 1e-6, 1 - 1e-6)
        assert predict.shape[0] == target.shape[
            0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        return binary_cross_entropyloss(predict, target)


class CELoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(CELoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        predict = torch.softmax(predict, dim=1)
        assert predict.shape == target.shape, 'predict & target shape do not match'
        criteon = BinaryCrossEntropyLoss(**self.kwargs)
        avg_loss = 0
        # print(predict.max(), predict.min(), target.max(), target.min())
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                loss = criteon(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert len(self.weight) == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    loss *= self.weight[i]
                avg_loss += loss
        return avg_loss


class SingleDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(SingleDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        intersection = (predict * target).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (predict.sum() + target.sum() + self.smooth))
        # print(torch.sum(predict), torch.sum(target))
        return loss


class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        predict = torch.softmax(predict, dim=1)
        assert predict.shape == target.shape, 'predict & target shape do not match'
        criteon = SingleDiceLoss()
        avg_loss = 0
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                loss = criteon(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert len(self.weight) == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    loss *= self.weight[i]
                avg_loss += loss
        return avg_loss


class CEDiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None):
        super(CEDiceLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        predict = torch.softmax(predict, dim=1)
        assert predict.shape == target.shape, 'predict & target shape do not match'
        criteon_dice = SingleDiceLoss()
        criteon_ce = BinaryCrossEntropyLoss()
        avg_loss = 0
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                loss = criteon_ce(predict[:, i], target[:, i]) + criteon_dice(
                    predict[:, i], target[:, i])
                if self.weight is not None:
                    assert len(self.weight) == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    loss *= self.weight[i]
                avg_loss += loss
        return avg_loss


def loss_kd(outputs, labels, teacher_outputs, alpha, T):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss