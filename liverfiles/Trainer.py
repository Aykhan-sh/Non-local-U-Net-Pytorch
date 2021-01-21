import torch
import wandb
import os
from .utils import get_lr

class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, metric_counter, train_dl, val_dl, device,
                 project_name, run_name, hparams=None):
        """
        :param model: torch model
        :param optimizer: torch optimizer
        :param scheduler: torch scheduler
        :param criterion: Loss function
        :param metric_counter: Utils.MetricCounter instance
        :param train_dl: Train dataloader
        :param val_dl: Validation Dataloader
        :param device: string. 'cuda' or 'cpu'
        :param root: directory to save logs and weights  #FIXME
        """
        root = "weights"
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device
        self.metric_counter = metric_counter
        self.metric_history = []
        self.best_metrics = None
        self.best_weights_path = None
        self.hparams = {
            'batch_size': train_dl.batch_size,
            'criterion': type(criterion).__name__,
            'optimzer': type(optimizer).__name__,
        }
        if hparams is not None:
            self.hparams.update(hparams)
        wandb.init(project=project_name, name=run_name, config=self.hparams)
        weight_path = os.path.join(wandb.run.dir[:-5], root)
        os.makedirs(weight_path, exist_ok=True)
        self.root = weight_path

    def get_lr(self):
        return get_lr(self.optimizer)

    def get_target(self):
        """
        :return: string. name of target metric
        """
        return self.metric_counter.target_metric

    def get_best_target(self) -> Number:
        """
        :return: measure of the best target metric
        """
        if self.best_metrics is not None:
            return self.best_metrics[self.get_target()]

    def prepare_input(self, x, labels):
        """
        Preprocessing of the output from dataloader before predictions are made.
        :param x: first output of the dataloader
        :param labels: second parameters of the dataloader- ground truth. If None then dataloader in inference mode
        :return: preprocessed x and labels or single x
        """
        x = x.to(self.device).float()
        if labels is not None:
            labels = labels.to(self.device).long()
            return x, labels
        return x

    def optimizer_step(self, x, labels):
        """
        :param x: first output of prepare_input function
        :param labels: second output of prepare_input function
        :return: model output and loss function of the step
        """
        self.optimizer.zero_grad()
        preds = self.model(x)
        loss_values = self.criterion(preds, labels)
        loss_values.backward()
        self.optimizer.step()
        return preds, loss_values

    def scheduler_step(self, metrics_list):
        """
        :param metrics: metrics of the step counted by MetricCounter
        :return: None
        """
        metrics = metrics_list[0]
        self.scheduler.step(metrics[self.metric_counter.target_metric])

    def log(self, metrics_list, epoch):
        cm = "confusion_matrix"
        metrics_to_log = {}
        for metrics in metrics_list:  # getting only scalars
            metrics_to_log.update({metrics["mode"] + " " + k: v for k, v in metrics.items() if k not in ["mode", cm]})
            if cm in metrics:
                metrics_to_log[metrics['mode'] + ' ' + cm] = [
                    wandb.Image(confusion_matrix_image(metrics[cm]), mode="RGB")]
        metrics_to_log['lr'] = self.get_lr()
        wandb.log(metrics_to_log, step=epoch)

    @staticmethod
    def data_to_list(preds, labels) -> Tuple[list, list]:
        """
        Override this method to postprocess data. Output must be list.
        Default implementation takes argmax from preds and detaches both input and converts it to list.
        :param preds: first output from optimizer_step function
        :param labels: second output from prepare_input function
        :return: processed preds and labels to list
        """
        post_preds = preds.detach().cpu().numpy().tolist()
        if labels is not None:
            post_gt = labels.detach().cpu().numpy().tolist()
            return post_preds, post_gt
        return post_preds

    def train_one_epoch(self):
        t = tqdm(enumerate(self.train_dl), total=len(self.train_dl), desc='Train', leave=False)
        result, gt = [], []
        loss_sum = 0
        self.model.train()
        for idx, (x, labels) in t:
            x, labels = self.prepare_input(x, labels)
            preds, loss_values = self.optimizer_step(x, labels)
            preds, labels = self.data_to_list(preds, labels)
            result += preds
            gt += labels
            loss_sum += loss_values.item()
            t.set_postfix(loss=f'{loss_sum / (idx + 1):.3f}')
            t.update()
        metrics = self.metric_counter(gt, result, {'loss': loss_sum / len(self.train_dl), "mode": "Train"})
        return metrics

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        t = tqdm(enumerate(self.val_dl), total=len(self.val_dl), desc='Val', leave=False)
        loss_sum = 0
        result, gt = [], []
        for idx, (x, labels) in t:
            x, labels = self.prepare_input(x, labels)
            preds = self.model(x)
            loss_sum += self.criterion(preds, labels).item()
            preds, labels = self.data_to_list(preds, labels)
            result += preds
            gt += labels
        metrics = self.metric_counter(gt, result, {'loss': loss_sum / len(self.train_dl), "mode": "Val"})
        return metrics

    @torch.no_grad()
    def infer(self, dl):
        t = tqdm(enumerate(dl), total=len(dl), desc='Test', leave=False)
        result = []
        for idx, x in t:
            x = self.prepare_input(x, None)
            preds = self.model(x)
            preds = self.data_to_list(preds, None)
            result += preds
        return result

    def save(self, save_tuple, metric: dict, mode='all', other={}, ceil=4, filename=None):
        """
        :param save_tuple: (model, optimizer, scheduler, epoch)
        :param metric: metric of current epoch
        :param mode: string 'all' or 'best'
        :param other: dictionary of other parameters to save
        :param ceil: number to ceil metric in file name
        :param filename: optional. string name for the ckp file
        :return: None. Saves cfg to directory
        """
        save = False
        best = False

        if self.best_metrics is None:  # if first epoch
            save = True
            best = True
            self.best_metrics = metric
        else:  # if epoch is not the first
            if self.metric_counter.compare_target(metric, self.best_metrics):
                save = True
                best = True
                self.best_metrics = metric
        if mode == 'all':
            save = True
        if save:
            model, optimizer, scheduler, epoch = save_tuple
            if filename is None:
                filename = f'ep{epoch}_m{metric[self.get_target()]:.{ceil}}.ckp'
            filepath = os.path.join(self.root, filename)
            filepath = path_uniquify(filepath)
            torch.save({
                'epoch': epoch,
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'metrics': metric,
                **other},
                filepath)
            if best:
                self.best_weights_path = filepath

    def train(self, epochs, save_kw={}, ckp_epoch=0):
        """
        :param epochs: number of epochs to train
        :param save_kw: string. kwargs of self.save function
        :param ckp_epoch:
        :return:
        """

        t = tqdm(range(ckp_epoch, epochs), total=epochs, initial=ckp_epoch, desc='Epoch')
        save_kw['other'] = {'all_epochs': epochs}
        for epoch in t:
            try:
                train_metrics = self.train_one_epoch()
                val_metrics = self.validate()
                self.log([train_metrics, val_metrics], epoch)
                self.save((self.model, self.optimizer, self.scheduler, epoch), val_metrics, **save_kw)
                self.scheduler_step([train_metrics, val_metrics])
            except KeyboardInterrupt:
                if epoch > ckp_epoch:  # ensure that at least one epoch was covered.
                    save_kw['mode'] = 'all'
                    self.save((self.model, self.optimizer, self.scheduler, epoch), val_metrics, **save_kw,
                              filename='interrupted.ckp')
                    print('Model has been saved')
                raise KeyboardInterrupt
        return val_metrics

    def load_model(self, filepath=None):
        """
        :param filepath: string or None. path to ckp file. if None, best_weights_path are used
        :return: None. Loads weighs
        """
        if filepath is None:
            filepath = self.best_weights_path
        self.model.load_state_dict(torch.load(filepath)['weights'])

    def uptrain(self, checkpoint, train_kw=None):
        """
        :param checkpoint: path to saved ckp file
        :param train_kw: kwargs for train function
        :return: test metrics
        """
        if train_kw is None:
            train_kw = {}
        ckp = torch.load(checkpoint)
        self.model.load_state_dict(ckp['weights'])
        self.optimizer.load_state_dict(ckp['optimizer'])
        self.scheduler.load_state_dict(ckp['scheduler'])
        train_kw['ckp_epoch'] = ckp['epoch']
        train_kw['epochs'] = ckp['all_epochs']
        return self.train(**train_kw)
