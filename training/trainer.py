import logging
import torch
import warnings
from typing import Optional, Dict, Tuple, List
import time
from torch.utils.data import DataLoader
from pathlib import Path
from utils.io_utils import save_to_json
from utils.config import get_train_config
import matplotlib.pyplot as plt


class SMNetTrainer:
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim,
                 logger_kwargs: Dict, device: Optional[str] = None, experiment_name: Optional[str] = "test"):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger_kwargs = logger_kwargs
        self.device = self._get_device(device)
        self.experiment_name = experiment_name
        self.model.to(self.device)

        # attributes
        self.train_loss_ = []
        self.val_loss_ = []
        logging.basicConfig(level=logging.INFO)

    def fit(self, train_loader, val_loader, epochs):
        logging.info(
            f"""Used device: {self.device} """
        )
        # track total training time
        total_start_time = time.time()
        # training
        for epoch in range(epochs):
            # track epoch time
            epoch_start_time = time.time()

            # train
            tr_loss = self._train(train_loader)

            # validate
            val_loss = self._validate(val_loader)

            self.train_loss_.append(tr_loss)
            self.val_loss_.append(val_loss)

            epoch_time = time.time() - epoch_start_time
            self._logger(
                tr_loss,
                val_loss,
                epoch + 1,
                epochs,
                epoch_time,
                **self.logger_kwargs
            )

        total_time = time.time() - total_start_time

        # final message
        logging.info(
            f"""End of training. Total time: {round(total_time, 5)} seconds"""
        )

    def _train(self, dataloader: DataLoader):
        self.model.train()
        loss = 0.0
        for cur_data in dataloader:
            # move to device
            cur_data = self._to_device(cur_data, device=self.device)
            cur_img_batch, cur_trans_img_batch, cur_gt_match_batch, cur_gt_trans_batch = cur_data
            # forward pass
            prediction = self.model(cur_img_batch, cur_trans_img_batch)

            # loss
            loss = self._compute_combined_loss(prediction, (cur_gt_match_batch, cur_gt_trans_batch))

            # remove gradient from previous passes
            self.optimizer.zero_grad()

            # backprop
            loss.backward()

            # parameters update
            self.optimizer.step()

        return loss.item()

    def _validate(self, dataloader):
        self.model.eval()

        with torch.no_grad():
            for cur_data in dataloader:
                # move to device
                cur_data = self._to_device(cur_data, device=self.device)
                cur_img_batch, cur_trans_img_batch, cur_gt_match_batch, cur_gt_trans_batch = cur_data
                prediction = self.model(cur_img_batch, cur_trans_img_batch)
                loss = self._compute_combined_loss(prediction, (cur_gt_match_batch, cur_gt_trans_batch))

        return loss.item()

    def _compute_combined_loss(self, pred: Tuple, gt: Tuple):
        # model returns matching probability and transform prediction.
        combined_loss = self.criterion(pred, gt)
        return combined_loss

    def _logger(self, tr_loss, val_loss, epoch, epochs, epoch_time, show=True, update_step=20):
        if show:
            if epoch % update_step == 0 or epoch == 1:
                # to satisfy pep8 common limit of characters
                msg = f"Epoch {epoch}/{epochs} | Train loss: {tr_loss}"
                msg = f"{msg} | Validation loss: {val_loss}"
                msg = f"{msg} | Time/epoch: {round(epoch_time, 5)} seconds"
                logging.info(msg)

    def _to_device(self, tensors: Tuple[torch.Tensor], device) -> List[torch.Tensor]:
        tensors = list(tensors)
        tensors = list(map(lambda x: x.to(device), tensors))
        return tensors

    def _get_device(self, device) -> str:
        if device is None:
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            msg = f"Device was automatically selected: {dev}"
            warnings.warn(msg)
        else:
            dev = device

        return dev

    def save_model(self, path: Path):
        if not path.parent.exists():
            path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def save_experiment(self, experiments_dir: Path):
        if not experiments_dir.exists():
            experiments_dir.mkdir(parents=True, exist_ok=True)

        cur_experiment_dir = experiments_dir / self.experiment_name
        cur_experiment_dir.mkdir(parents=True, exist_ok=True)
        current_training_config = get_train_config()
        save_to_json(current_training_config, str(cur_experiment_dir / "_train_config.json"))

        # save train-val loss graph
        # Plotting the training loss
        plt.subplot(1, 2, 1)  # Create a subplot for the first plot (training loss)
        plt.plot(self.train_loss_, 'r-', label='Training Loss')  # 'r-' denotes red color and line style
        plt.title('Training Loss')  # Set the title for the training loss plot
        plt.xlabel('Epochs')  # Label for the x-axis
        plt.ylabel('Loss')  # Label for the y-axis
        plt.legend()  # Show the legend

        # Plotting the validation loss
        plt.subplot(1, 2, 2)  # Create a subplot for the second plot (validation loss)
        plt.plot(self.val_loss_, 'b-', label='Validation Loss')  # 'b-' denotes blue color and line style
        plt.title('Validation Loss')  # Set the title for the validation loss plot
        plt.xlabel('Epochs')  # Label for the x-axis
        plt.ylabel('Loss')  # Label for the y-axis
        plt.legend()  # Show the legend

        # Display the plot
        plt.tight_layout()  # Adjust the layout to avoid overlapping labels
        plt.savefig(str(cur_experiment_dir / "loss_plot.png"))
