import sys
import torch
from tqdm.autonotebook import tqdm as tqdm
from .meter import Meter


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
        if self.loss:
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
        loss_meter = Meter()
        metrics_meters = {m.__name__: Meter(m.resolve if m.is_micro else None) for m in self.metrics}

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
            leave=not (self.verbose)
        ) as iterator:
            for x, y, fname in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y, fname)

                # update loss logs
                if loss:
                    loss_meter.add(loss)
                    logs.update({self.loss.__name__: loss_meter.value()})

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y)
                    metrics_meters[metric_fn.__name__].add(metric_value)
                logs.update({k: v.value() for k, v in metrics_meters.items()})

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
            loss = self.loss(prediction, y) if self.loss else None
        return loss, prediction
