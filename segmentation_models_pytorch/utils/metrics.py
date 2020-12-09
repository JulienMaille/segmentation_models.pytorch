from . import base
from . import functional as F
from .base import Activation
import torch

class Iou(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, mask=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.mask = mask

    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            mask=self.mask
        )


class MicroIou(base.Metric):

    __name__ = 'µ-iou'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, mask=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.is_micro = True
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.mask = mask

    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.micro_iou(
            y_pr, y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            mask=self.mask
        )

    def resolve(self, micro):
        return (micro[0]+self.eps)/(micro[1]+self.eps)


class Fscore(base.Metric):

    @property
    def __name__(self):
        return 'fscore_β:{:.1f}'.format(self.beta)

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, mask=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.mask = mask

    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.f_score(
            y_pr, y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            mask=self.mask
        )


class MicroFscore(base.Metric):

    @property
    def __name__(self):
        return self.build_name(self.beta)

    def name(self, i):
        if isinstance(self.beta, tuple):
            return self.build_name(self.beta[i])
        else:
            return __name__

    def build_name(self, beta):
        if isinstance(beta, tuple):
            return 'µ-fscore_β:' + '/'.join(str(x) for x in beta)
        else:
            return 'µ-fscore_β:' + str(beta)

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, mask=None, **kwargs):
        super().__init__(**kwargs)
        self.is_micro = True
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.mask = mask


    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.micro_f_score(
            y_pr, y_gt,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            mask=self.mask
        )

    def resolve(self, micro):
        return (micro[0]+self.eps)/(micro[1]+self.eps)


class Accuracy(base.Metric):

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.accuracy(
            y_pr, y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Recall(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.recall(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class MicroRecall(base.Metric):

    __name__ = 'µ-recall'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.is_micro = True
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels


    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.micro_recall(
            y_pr, y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

    def resolve(self, micro):
        return (micro[0]+self.eps)/(micro[1]+self.eps)


class Precision(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.precision(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class MicroPrecision(base.Metric):

    __name__ = 'µ-precision'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.is_micro = True
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.micro_precision(
            y_pr, y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

    def resolve(self, micro):
        return (micro[0]+self.eps)/(micro[1]+self.eps)