import torch


def _ignore_last_mask(pr, gt, ignoreLastMask=False):
    if ignoreLastMask:
        pr = pr * (1.-gt[:,-1:,:,:])
        gt = gt[:,:-1,:,:]
    return pr, gt

def _take_channels(*xs, keep_channels=None):
    if keep_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel in keep_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def micro_iou(pr, gt, threshold=None, keep_channels=None, ignoreLastMask=False):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr, gt = _ignore_last_mask(pr, gt, ignoreLastMask=ignoreLastMask)
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, keep_channels=keep_channels)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection
    return intersection, union

def iou(pr, gt, eps=1e-7, threshold=None, keep_channels=None, ignoreLastMask=False):
    i,u = micro_iou(pr, gt, threshold, keep_channels, ignoreLastMask)
    return (i+eps) / (u+eps)
jaccard = iou


def one_chan_f_score(pr, gt, beta):
    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp
    return (1 + beta**2) * tp, (1 + beta**2) * tp + beta**2 * fn + fp


def micro_f_score(pr, gt, beta=1, threshold=None, keep_channels=None, ignoreLastMask=False):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr, gt = _ignore_last_mask(pr, gt, ignoreLastMask=ignoreLastMask)
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, keep_channels=keep_channels)

    if isinstance(beta, tuple) and keep_channels:
        beta = [b for idx, b in enumerate(beta) if idx in keep_channels]
        if len(beta) == 1 : beta = beta[0]
    if isinstance(beta, tuple):
        i = u = 0
        for idx, b in enumerate(beta):
            ic, uc = one_chan_f_score(*_take_channels(pr, gt, keep_channels=[idx]), b)
            i += ic
            u += uc
        return i, u
    else:
        return one_chan_f_score(pr, gt, beta)


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, keep_channels=None, ignoreLastMask=False):
    i, u = micro_f_score(pr, gt, beta, threshold, keep_channels, ignoreLastMask)
    return (i+eps) / (u+eps)


def accuracy(pr, gt, threshold=0.5, keep_channels=None, ignoreLastMask=False):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr, gt = _ignore_last_mask(pr, gt, ignoreLastMask=ignoreLastMask)
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, keep_channels=keep_channels)

    tp = torch.sum(gt == pr, dtype=pr.dtype)
    if gt.is_contiguous():
        score = tp / gt.view(-1).shape[0]
    else:
        score = tp / gt.reshape(-1).shape[0]
    return score


def micro_precision(pr, gt, threshold=None, keep_channels=None, ignoreLastMask=False):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr, gt = _ignore_last_mask(pr, gt, ignoreLastMask=ignoreLastMask)
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, keep_channels=keep_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    return tp, tp+fp

def precision(pr, gt, eps=1e-7, threshold=None, keep_channels=None, ignoreLastMask=False):
    tp, prp = micro_iou(pr, gt, threshold, keep_channels, ignoreLastMask)
    return (tp+eps) / (prp+eps)


def micro_recall(pr, gt, threshold=None, keep_channels=None, ignoreLastMask=False):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """

    pr, gt = _ignore_last_mask(pr, gt, ignoreLastMask=ignoreLastMask)
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, keep_channels=keep_channels)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    return tp, tp+fn

def recall(pr, gt, eps=1e-7, threshold=None, keep_channels=None, ignoreLastMask=False):
    tp, gtp = micro_recall(pr, gt, threshold, keep_channels, ignoreLastMask)
    return (tp+eps) / (gtp+eps)
