import wandb
import os
from liverfiles.utils import get_lr, path_uniquify, unsplit_binary_mask
from liverfiles.metrics import *
from typing import Tuple
from numbers import Number
from tqdm.notebook import tqdm


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, train_dl, val_dl, device,
                 project_name, run_name, hparams=None):
        """
        :param model: torch model
        :param optimizer: torch optimizer
        :param scheduler: torch scheduler
        :param criterion: Loss function
        :param train_dl: Train dataloader
        :param val_dl: Validation Dataloader
        :param device: string. 'cuda' or 'cpu'
        :param root: directory to save logs and weights  #FIXME
        """
        root = "weights"
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device
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
        self.shape = self.train_dl.shape

    def get_lr(self):
        return get_lr(self.optimizer)

    def preprocess_input(self, x, labels):
        """
        Preprocessing of the output from dataloader before predictions are made.
        :param x: first output of the dataloader
        :param labels: second parameters of the dataloader- ground truth. If None then dataloader in inference mode
        :return: preprocessed x and labels or single x
        """
        x = x.to(self.device).float()
        if labels is not None:
            labels = labels.to(self.device).float()
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

    @staticmethod
    def postprocess_output(preds, labels) -> Tuple[list, list]:
        """
        Override this method to postprocess data. Output must be list.
        Default implementation takes argmax from preds and detaches both input and converts it to list.
        :param preds: first output from optimizer_step function
        :param labels: second output from prepare_input function
        :return: processed preds and labels to list
        """
        threshold = 0.5
        post_preds = to_numpy(preds)
        post_preds = (post_preds > threshold).astype('uint8')
        if labels is not None:
            post_gt = to_numpy(labels)
            return post_preds, post_gt
        return post_preds

    def scheduler_step(self, metrics):
        """
        :param metrics: metrics of the step counted by MetricCounter
        :return: None
        """
        self.scheduler.step(metrics['Val Loss'])

    @staticmethod
    def wb_mask(img, pred_mask, true_mask):
        labels = {
            0: 'healthy',
            1: 'primary',
            2: 'secondary'
        }
        return wandb.Image(img, masks={
            "prediction": {"mask_data": pred_mask, "class_labels": labels},
            "ground truth": {"mask_data": true_mask, "class_labels": labels}})

    def log(self, metrics, x, preds, labels, epoch):
        # metrics logging
        metrics_to_log = {}
        labels_name = ['primary', 'secondary']
        for key, value in metrics.items():
            if type(value) not in [float, int]:
                for idx, l in enumerate(labels_name):
                    metrics_to_log[f"{key} {l}"] = value[idx]
            else:
                metrics_to_log[key] = value
        wandb.log(metrics_to_log, step=epoch)
        # logging images
        img_size = x.shape[-2:]
        center = x.shape[2] // 2
        examples = x[:, :, center, :, :]
        pred_masks = preds[:, :, center, :, :]
        pred_masks = unsplit_binary_mask(pred_masks)
        gt_masks = labels[:, :, center, :, :]
        gt_masks = unsplit_binary_mask(gt_masks)
        wandb_images = []
        for i in range(x.shape[0]):
            wandb_images.append(self.wb_mask(examples[i], pred_masks[i], gt_masks[i]))
        wandb.log({'Images': wandb_images}, step=epoch)

    def train_one_epoch(self):
        t = tqdm(enumerate(self.train_dl), total=len(self.train_dl), desc='Train', leave=False)
        metrics = None
        result, gt = [], []
        loss_sum = 0
        self.model.train()
        for idx, (x, labels) in t:
            x, labels = self.preprocess_input(x, labels)
            preds, loss_values = self.optimizer_step(x, labels)
            preds, labels = self.postprocess_output(preds, labels)
            temp_metrics = count_metrics(labels, preds, "Train")
            if metrics is None:  # if the first iteration:
                metrics = temp_metrics
            else:  # sum all metrics
                metrics = sum_metrics(metrics, temp_metrics)
            loss_sum += loss_values.item()
            t.set_postfix(loss=f'{loss_sum / (idx + 1):.3f}')
            t.update()
        metrics = divide_metrics(metrics, len(self.train_dl))  # averaging metrics
        metrics['Train Loss'] = loss_sum / len(self.train_dl)  # adding to metric dictionary loss value
        return metrics

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        t = tqdm(enumerate(self.val_dl), total=len(self.val_dl), desc='Val', leave=False)
        loss_sum = 0
        metrics = None
        for idx, (x, labels) in t:
            x, labels = self.preprocess_input(x, labels)
            preds = self.model(x)
            loss_sum += self.criterion(preds, labels).item()
            preds, labels = self.postprocess_output(preds, labels)
            temp_metrics = count_metrics(labels, preds, "Val")
            if metrics is None:  # if the first iteration:
                metrics = temp_metrics
            else:  # sum all metrics
                metrics = sum_metrics(metrics, temp_metrics)
        x = to_numpy(x)
        metrics = divide_metrics(metrics, len(self.val_dl))  # averaging metrics
        metrics['Val Loss'] = loss_sum / len(self.train_dl)  # adding to metric dictionary loss value
        return metrics, x, preds, labels

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
        model, optimizer, scheduler, epoch = save_tuple
        if filename is None:
            filename = f'ep{epoch}.ckp'
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
                metrics = self.train_one_epoch()
                val_metrics, x, preds, labels = self.validate()
                metrics.update(val_metrics)
                self.log(metrics, x, preds, labels, epoch)
                # self.save((self.model, self.optimizer, self.scheduler, epoch), metrics, **save_kw)
                self.scheduler_step(metrics)
            except KeyboardInterrupt:
                pass
            #     if epoch > ckp_epoch:  # ensure that at least one epoch was covered.
            #         save_kw['mode'] = 'all'
            #         self.save((self.model, self.optimizer, self.scheduler, epoch), metrics, **save_kw,
            #                   filename='interrupted.ckp')
            #         print('Model has been saved')
            #     raise KeyboardInterrupt
        return metrics

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
