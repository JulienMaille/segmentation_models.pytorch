import torch


def _take_channels(*xs, ignore_channels=None, mask=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def micro_iou(pr, gt, threshold=None, ignore_channels=None, mask=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    if mask is not None:
        pr = pr * mask
        gt = gt * mask
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection
    return intersection, union

def iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None, mask=None):
    i,u = micro_iou(pr, gt, threshold, ignore_channels, mask)
    return (i+eps) / (u+eps)
jaccard = iou


def one_chan_f_score(pr, gt, beta):
    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp
    return (1 + beta ** 2) * tp, (1 + beta ** 2) * tp + beta ** 2 * fn + fp


def micro_f_score(pr, gt, beta=1, threshold=None, ignore_channels=None, mask=None):
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

    if mask is not None:
        pr = pr * mask
        gt = gt * mask
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    if isinstance(beta, tuple):
        i = u = 0
        for idx, b in enumerate(beta):
            ic, uc = one_chan_f_score(pr[:,idx,:,:], gt[:,idx,:,:], b)
            i += ic
            u += uc
        return i, u
    else:
        return one_chan_f_score(pr, gt, beta)


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None, mask=None):
    i, u = micro_f_score(pr, gt, beta, threshold, ignore_channels, mask)
    return (i+eps) / (u+eps)


def accuracy(pr, gt, threshold=0.5, ignore_channels=None):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt == pr, dtype=pr.dtype)
    if gt.is_contiguous():
        score = tp / gt.view(-1).shape[0]
    else:
        score = tp / gt.reshape(-1).shape[0]
    return score


def micro_precision(pr, gt, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    return tp, tp+fp

def precision(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    tp, prp = micro_iou(pr, gt, threshold, ignore_channels)
    return (tp+eps) / (prp+eps)


def micro_recall(pr, gt, threshold=None, ignore_channels=None):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    return tp, tp+fn

def recall(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    tp, gtp = micro_recall(pr, gt, threshold, ignore_channels)
    return (tp+eps) / (gtp+eps)
