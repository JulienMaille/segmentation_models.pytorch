import sys
import torch
from tqdm.autonotebook import tqdm as tqdm
from .meter import AverageValueMeter

class Epoch:

    def __init__(self, model, loss, metrics, stage_name, nb_classes, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.nb_classes = nb_classes
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        if self.loss is not None:
            self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{}: {:.3}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter(metric.resolve if metric.is_micro else None) for metric in self.metrics}
        if self.nb_classes > 1:
            metrics_class_meters = {metric.__name__ + '_c{}'.format(cls): AverageValueMeter(metric.resolve if metric.is_micro else None)
                for metric in self.metrics for cls in range(self.nb_classes)}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose), leave=not (self.verbose)) as iterator:
            for x, y, fname in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y, fname)

                # update loss logs
                if loss is not None:
                    loss_value = loss.cpu().detach().numpy()
                    loss_meter.add(loss_value)
                    loss_logs = {self.loss.__name__: loss_meter.value()}
                    logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_total = metric_fn(y_pred, y)
                    metrics_meters[metric_fn.__name__].add(metric_total)
                metrics_logs = {k: v.value() for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.nb_classes > 1:
                    for metric_fn in self.metrics:
                        for cls in range(self.nb_classes):
                            # erase all but c channel
                            sel = list(range(y_pred.shape[1]))
                            sel.remove(cls)
                            y_c, y_pred_c = y.clone(), y_pred.clone()
                            y_c[:, sel, :, :] = 0
                            y_pred_c[:, sel, :, :] = 0
                            metric_class = metric_fn(y_pred_c, y_c)
                            metrics_class_meters['{}_c{}'.format(metric_fn.__name__, cls)].add(metric_class)
                    metrics_class_logs = {k: v.value() for k, v in metrics_class_meters.items()}
                    logs.update(metrics_class_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, nb_classes, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            nb_classes=nb_classes,
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, fname):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, nb_classes, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            nb_classes=nb_classes,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y) if self.loss is not None else None
        return loss, prediction
