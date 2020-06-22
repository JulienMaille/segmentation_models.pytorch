from . import base
from . import functional as F
from .base import Activation
import numpy as np

class Iou(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, mask=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.mask = mask

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            mask=self.mask
        ).cpu().detach().numpy()


class MicroIou(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, mask=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.is_micro = True
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.mask = mask

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        i,u = F.micro_iou(
            y_pr, y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            mask=self.mask
        )
        return np.array([float(i.cpu().detach()), float(u.cpu().detach())])

    def resolve(self, micro):
        return (micro[0]+self.eps)/(micro[1]+self.eps)


class Fscore(base.Metric):

    @property
    def __name__(self):
        if self.beta is 1.:
            return 'f_score'
        else:
            return 'f_score_β:{:.1f}'.format(self.beta)

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, mask=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.mask = mask

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.f_score(
            y_pr, y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            mask=self.mask
        ).cpu().detach().numpy()


class MicroFscore(base.Metric):

    @property
    def __name__(self):
        if self.beta is 1.:
            return 'micro_f_score'
        else:
            return 'micro_f_score_β:{:.1f}'.format(self.beta)

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, mask=None, **kwargs):
        super().__init__(**kwargs)
        self.is_micro = True
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.mask = mask

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        i,u = F.micro_f_score(
            y_pr, y_gt,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            mask=self.mask
        )
        return np.array([float(i.cpu().detach()), float(u.cpu().detach())])

    def resolve(self, micro):
        return (micro[0]+self.eps)/(micro[1]+self.eps)


class Accuracy(base.Metric):

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.accuracy(
            y_pr, y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        ).cpu().detach().numpy()


class Recall(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.recall(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        ).cpu().detach().numpy()


class MicroRecall(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.is_micro = True
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        tp, p = F.micro_recall(
            y_pr, y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
        return np.array([float(tp.cpu().detach()), float(p.cpu().detach())])

    def resolve(self, micro):
        return (micro[0]+self.eps)/(micro[1]+self.eps)


class Precision(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.precision(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        ).cpu().detach().numpy()


class MicroPrecision(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.is_micro = True
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        tp, p = F.micro_precision(
            y_pr, y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
        return np.array([float(tp.cpu().detach()), float(p.cpu().detach())])

    def resolve(self, micro):
        return (micro[0]+self.eps)/(micro[1]+self.eps)