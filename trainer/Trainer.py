from trainer.metrics import *
from typing import Tuple
from tqdm.notebook import tqdm
from trainer.utils import *
from nonlocalunet.infer import infer


class Trainer:
    def __init__(self, model, num_classes, shape, optimizer, scheduler, criterion, train_dl, val_dl, test_ids, device,
                 run_name, hparams=None, window=None, root='weights'):
        """
        :param model: torch model
        :param num_classes: number of classes
        :param optimizer: torch optimizer
        :param scheduler: torch scheduler
        :param criterion: Loss function
        :param train_dl: Train dataloader
        :param val_dl: Validation Dataloader
        :param device: string. 'cuda' or 'cpu'
        :param root: directory to save logs and weights  #FIXME
        """
        root = "weights"
        self.model = model
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_ids = test_ids
        self.device = device
        self.metric_history = []
        self.best_metrics = None
        self.best_weights_path = None
        self.hparams = {
            'batch_size': train_dl.batch_size,
            'criterion': type(criterion).__name__,
            'optimzer': type(optimizer).__name__,
        }
        self.logger = create_logger(run_name)
        self.root = os.path.join(self.logger.log_dir, root)
        if hparams is not None:
            self.hparams.update(hparams)
        self.shape = shape
        self.current_epoch = None
        if window is None:
            self.window = self.shape

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
        preds = to_numpy(preds)
        post_preds = preds.copy()
        post_preds = (post_preds > threshold).astype('uint8')
        if labels is not None:
            post_gt = to_numpy(labels)
            return post_preds, preds, post_gt
        return post_preds, preds

    def scheduler_step(self, metrics):
        """
        :param metrics: metrics of the step counted by MetricCounter
        :return: None
        """
        self.scheduler.step(metrics['Val Loss'])

    def log(self, metric_dict):
        for metric in metric_dict:
            if metric in ['Train Loss', 'Val Loss', 'Lr']:
                continue
            for l in metric_dict[metric]:
                self.logger.add_scalar(f"{metric}/{l}", metric_dict[metric][l], self.current_epoch)
        self.logger.add_scalar('Loss/Train', metric_dict['Train Loss'], self.current_epoch)
        self.logger.add_scalar('Loss/Val', metric_dict['Val Loss'], self.current_epoch)
        self.logger.add_scalar('Lr', self.get_lr(), self.current_epoch)

    def log_video(self, x, preds, labels, name):
        x = x.swapaxes(1, 2)
        preds = preds.swapaxes(1, 2)
        labels = labels.swapaxes(1, 2)
        for i in range(self.num_classes):
            img_to_log = img_with_masks(x, [preds[:, :, [i], :, :],
                                            labels[:, :, [i], :, :] > 0.5], 0.4)
            self.logger.add_video(f'{name}/' + label_names[i], img_to_log, self.current_epoch)

    def train_one_epoch(self):
        t = tqdm(self.train_dl, total=len(self.train_dl), desc='Train', leave=False)
        metrics = None
        loss_sum = 0
        idx = 0  # for some reason len(train_dl is not working)
        self.model.train()
        for x, labels in t:
            idx += 1
            x, labels = self.preprocess_input(x, labels)
            raw_preds, loss_values = self.optimizer_step(x, labels)
            post_preds, raw_preds, labels = self.postprocess_output(raw_preds, labels)
            temp_metrics = count_metrics(labels, raw_preds, post_preds, 'Train')
            if metrics is None:  # if the first iteration:
                metrics = temp_metrics
            else:  # sum all metrics
                metrics = sum_metrics(metrics, temp_metrics)
            loss_sum += loss_values.item()
            t.set_postfix(loss=f'{loss_sum / idx:.3f}')
            t.update()
        metrics = divide_metrics(metrics, idx)  # averaging metrics
        metrics['Train Loss'] = loss_sum / len(self.train_dl)  # adding to metric dictionary loss value
        return metrics

    @torch.no_grad()
    def validate(self):
        t = tqdm(self.val_dl, total=len(self.val_dl), desc='Val', leave=False)
        metrics = None
        loss_sum = 0
        idx = 0  # for some reason len(train_dl is not working)
        self.model.eval()
        for x, labels in t:
            idx += 1
            x, labels = self.preprocess_input(x, labels)
            raw_preds = self.model(x)
            loss_sum += self.criterion(raw_preds, labels).item()
            post_preds, raw_preds, labels = self.postprocess_output(raw_preds, labels)
            temp_metrics = count_metrics(labels, raw_preds, post_preds, 'Val')
            if metrics is None:  # if the first iteration:
                metrics = temp_metrics
            else:  # sum all metrics
                metrics = sum_metrics(metrics, temp_metrics)
            t.set_postfix(loss=f'{loss_sum / idx:.3f}')
            t.update()
        metrics = divide_metrics(metrics, idx)  # averaging metrics
        metrics['Val Loss'] = loss_sum / len(self.train_dl)  # adding to metric dictionary loss value
        return metrics

    @torch.no_grad()
    def test(self):
        self.model.eval()
        t = tqdm(range(len(self.test_ids)), total=len(self.test_ids), desc='Val', leave=False)
        loss_sum = 0
        for idx in t:
            img = open_ct(self.test_ids[idx])
            mask = open_mask(self.test_ids[idx])
            # img
            img = np.expand_dims(img, axis=0)  # C, D, W, H
            # mask
            mask = split_mask(mask, num_of_classes=self.num_classes)  # C, D, W, H
            # peds
            raw_preds = self.infer(img)  # C, D, W, H
            img, raw_preds, mask = (np.expand_dims(j, axis=0) for j in [img, raw_preds, mask])  # B, C, D, W, H
            self.log_video(img, raw_preds, mask, f'Example {self.test_ids[idx]}')

    def infer(self, img):
        return infer(self.model, img, self.shape, self.num_classes, self.window,
                     batch_size=self.train_dl.batch_size, num_workers=self.train_dl.num_workers,
                     device=self.device)

    def save(self, filename=None):
        """
        :param filename: optional. string name for the ckp file
        :return: None. Saves cfg to directory
        """
        if filename is None:
            filename = f"{self.current_epoch}.ckp"
        filepath = os.path.join(self.root, filename)
        filepath = path_uniquify(filepath)
        self.logger.close()
        torch.save(self, filepath)

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
            self.current_epoch = epoch
            try:
                metrics = self.train_one_epoch()
                val_metrics = self.validate()
                metrics.update(val_metrics)
                self.log(metrics)
                # self.save()   #FIXME
                self.scheduler_step(metrics)
                if (epoch + 1) % 10 == 0:
                    self.test()
            except KeyboardInterrupt:
                pass
                if epoch > ckp_epoch:  # ensure that at least one epoch was covered.
                    self.save(filename='interrupted.ckp')
                    print('Model has been saved')
                raise KeyboardInterrupt
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
