import torch
import torch.nn as nn

from . import base
from . import functional as F
from ..base.modules import Activation


class JaccardLoss(base.Loss):

    def __init__(self, eps=1., activation=None, keep_channels=None, ignoreLastMask=False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.keep_channels = keep_channels
        self.ignoreLastMask = ignoreLastMask

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=None,
            keep_channels=self.keep_channels,
            ignoreLastMask=self.ignoreLastMask
        )


class DiceLoss(base.Loss):
    @property
    def __name__(self):
        if isinstance(self.beta, tuple):
            return 'dice_loss_β:' + '/'.join(str(x) for x in self.beta)
        else:
            return 'dice_loss_β:' + str(self.beta)

    def __init__(self, eps=1., beta=1., activation=None, keep_channels=None, ignoreLastMask=False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.keep_channels = keep_channels
        self.ignoreLastMask = ignoreLastMask

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            keep_channels=self.keep_channels,
            ignoreLastMask=self.ignoreLastMask
        )


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, beta=1., activation=None, keep_channels=None, mask=None, **kwargs):
        super().__init__(eps, beta, activation, keep_channels, mask, **kwargs)
        self.bce = nn.BCEWithLogitsLoss(weight=mask)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)

        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return 0.6*dice + 0.4*bce


class MultiLabelDiceLoss(DiceLoss):
    __name__ = 'mean_dice_loss'

    def __init__(self, eps=1., beta=1., activation=None, keep_channels=None, mask=None, **kwargs):
        super().__init__(eps, beta, activation, keep_channels, mask, **kwargs)

    def forward(self, y_pr, y_gt):
        losses = []
        for i in range(y_pr.shape[1]):
            losses.append(super().forward(y_pr[:,i,:,:].unsqueeze(1), y_gt[:,i,:,:].unsqueeze(1)))
        loss = torch.mean(torch.stack(losses))
        return loss

class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    def __init__(self, mask=None, **kwargs):
        super().__init__(**kwargs)
        self.bce = nn.BCELoss(reduction='mean', weight=mask)

    def forward(self, y_pr, y_gt):
        return self.bce(y_pr, y_gt)


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    def __init__(self, mask=None, **kwargs):
        super().__init__(**kwargs)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean', weight=mask)

    def forward(self, y_pr, y_gt):
        return self.bce(y_pr, y_gt)


class FocalLoss(base.Loss):
    def __init__(self, alpha=0.25, gamma=2, mask=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none', weight=mask)

    def forward(self, y_pr, y_gt):
        bce = self.bce(y_pr, y_gt)
        pt = torch.exp(-bce)  # prevents nans when probability 0
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


class FocalWithLogitsLoss(base.Loss):
    def __init__(self, alpha=0.25, gamma=2, mask=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none', weight=mask)

    def forward(self, y_pr, y_gt):
        bce = self.bce(y_pr, y_gt)
        pt = torch.exp(-bce)  # prevents nans when probability 0
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()
