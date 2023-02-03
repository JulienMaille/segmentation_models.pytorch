from . import base
from . import functional as F
from ..base.modules import Activation
import torch

class Iou(base.Metric):

    @property
    def __name__(self):
        name = 'iou'
        if self.keep_channels:
            name += '_c{}'.format(','.join(str(x) for x in self.keep_channels))
        return name

    def __init__(self, eps=1e-7, threshold=0.0, activation=None, keep_channels=None, ignoreLastMask=False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.keep_channels = keep_channels
        self.ignoreLastMask = ignoreLastMask

    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            keep_channels=self.keep_channels,
            ignoreLastMask=self.ignoreLastMask
        )


class MicroIou(base.Metric):

    @property
    def __name__(self):
        name = 'µ-iou'
        if self.keep_channels:
            name += '_c{}'.format(','.join(str(x) for x in self.keep_channels))
        return name

    def __init__(self, eps=1e-7, threshold=0.0, activation=None, keep_channels=None, ignoreLastMask=False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.is_micro = True
        self.threshold = threshold
        self.activation = Activation(activation)
        self.keep_channels = keep_channels
        self.ignoreLastMask = ignoreLastMask

    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.micro_iou(
            y_pr, y_gt,
            threshold=self.threshold,
            keep_channels=self.keep_channels,
            ignoreLastMask=self.ignoreLastMask
        )

    def resolve(self, micro):
        return (micro[0]+self.eps)/(micro[1]+self.eps)


class Fscore(base.Metric):

    @property
    def __name__(self):
        if isinstance(self.beta, tuple):
            name = 'fscore_β:' + '/'.join(str(x) for x in self.beta)
        else:
            name = 'fscore_β:' + str(self.beta)
        if self.keep_channels:
            name += '_c{}'.format(','.join(str(x) for x in self.keep_channels))
        return name

    def __init__(self, beta=1, eps=1e-7, threshold=0.0, activation=None, keep_channels=None, ignoreLastMask=False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.keep_channels = keep_channels
        self.ignoreLastMask = ignoreLastMask

    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.f_score(
            y_pr,
            y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            keep_channels=self.keep_channels,
            ignoreLastMask=self.ignoreLastMask
        )


class MicroFscore(base.Metric):

    @property
    def __name__(self):
        if isinstance(self.beta, tuple):
            name = 'µ-fscore_β:' + '/'.join(str(x) for x in self.beta)
        else:
            name = 'µ-fscore_β:' + str(self.beta)
        if self.keep_channels:
            name += '_c{}'.format(','.join(str(x) for x in self.keep_channels))
        return name

    def __init__(self, beta=1, eps=1e-7, threshold=0.0, activation=None, keep_channels=None, ignoreLastMask=False, **kwargs):
        super().__init__(**kwargs)
        self.is_micro = True
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.keep_channels = keep_channels
        self.ignoreLastMask = ignoreLastMask


    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.micro_f_score(
            y_pr, y_gt,
            beta=self.beta,
            threshold=self.threshold,
            keep_channels=self.keep_channels,
            ignoreLastMask=self.ignoreLastMask
        )

    def resolve(self, micro):
        return (micro[0]+self.eps)/(micro[1]+self.eps)


class Accuracy(base.Metric):

    @property
    def __name__(self):
        name = 'accuracy'
        if self.keep_channels:
            name += '_c{}'.format(','.join(str(x) for x in self.keep_channels))
        return name

    def __init__(self, threshold=0.0, activation=None, keep_channels=None, ignoreLastMask=False, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.keep_channels = keep_channels
        self.ignoreLastMask = ignoreLastMask

    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.accuracy(
            y_pr,
            y_gt,
            threshold=self.threshold,
            keep_channels=self.keep_channels,
            ignoreLastMask=self.ignoreLastMask,
        )


class Recall(base.Metric):

    @property
    def __name__(self):
        name = 'recall'
        if self.keep_channels:
            name += '_c{}'.format(','.join(str(x) for x in self.keep_channels))
        return name

    def __init__(self, eps=1e-7, threshold=0.0, activation=None, keep_channels=None, ignoreLastMask=False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.keep_channels = keep_channels
        self.ignoreLastMask = ignoreLastMask

    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.recall(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            keep_channels=self.keep_channels,
            ignoreLastMask=self.ignoreLastMask,
        )


class MicroRecall(base.Metric):

    @property
    def __name__(self):
        name = 'µ-recall'
        if self.keep_channels:
            name += '_c{}'.format(','.join(str(x) for x in self.keep_channels))
        return name

    def __init__(self, eps=1e-7, threshold=0.0, activation=None, keep_channels=None, ignoreLastMask=False, **kwargs):
        super().__init__(**kwargs)
        self.is_micro = True
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.keep_channels = keep_channels
        self.ignoreLastMask = ignoreLastMask


    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.micro_recall(
            y_pr, y_gt,
            threshold=self.threshold,
            keep_channels=self.keep_channels,
            ignoreLastMask=self.ignoreLastMask,
        )

    def resolve(self, micro):
        return (micro[0]+self.eps)/(micro[1]+self.eps)


class Precision(base.Metric):

    @property
    def __name__(self):
        name = 'precision'
        if self.keep_channels:
            name += '_c{}'.format(','.join(str(x) for x in self.keep_channels))
        return name

    def __init__(self, eps=1e-7, threshold=0.0, activation=None, keep_channels=None, ignoreLastMask=False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.keep_channels = keep_channels
        self.ignoreLastMask = ignoreLastMask

    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.precision(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            keep_channels=self.keep_channels,
            ignoreLastMask=self.ignoreLastMask,
        )


class MicroPrecision(base.Metric):

    @property
    def __name__(self):
        name = 'µ-precision'
        if self.keep_channels:
            name += '_c{}'.format(','.join(str(x) for x in self.keep_channels))
        return name

    def __init__(self, eps=1e-7, threshold=0.0, activation=None, keep_channels=None, ignoreLastMask=False, **kwargs):
        super().__init__(**kwargs)
        self.is_micro = True
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.keep_channels = keep_channels
        self.ignoreLastMask = ignoreLastMask

    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.micro_precision(
            y_pr, y_gt,
            threshold=self.threshold,
            keep_channels=self.keep_channels,
            ignoreLastMask=self.ignoreLastMask,
        )

    def resolve(self, micro):
        return (micro[0]+self.eps)/(micro[1]+self.eps)
