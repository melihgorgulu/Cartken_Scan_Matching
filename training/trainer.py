import logging
import torch
import warnings
from typing import Optional, Dict, Tuple, List
import time
from torch.utils.data import DataLoader
from pathlib import Path
from utils.io_utils import save_to_json, save_loss_graph, revert_image_transform, convert_tensor_to_pil, save_prediction_results
from utils.config import get_train_config
from training.early_stopping import EarlyStopper
import random
from PIL import Image



class SMNetTrainer:
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim,
                 logger_kwargs: Dict, train_stats_config: Dict, device: Optional[str] = None,
                 experiment_name: Optional[str] = "test", show_all_losses: bool = False, vis_predictions_every_n=None,
                 vis_validation=False):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger_kwargs = logger_kwargs
        self.show_all_losses = show_all_losses
        self.device = self._get_device(device)
        self.experiment_name = experiment_name
        self.train_stats_config = train_stats_config
        self.vis_predictions_every_n = vis_predictions_every_n
        self.vis_validation = vis_validation
        self.model.to(self.device)

        # attributes
        self.train_loss_: List[float] = []
        self.val_loss_: List[float] = []
        # also store the loss info values. Each loss info is a dictionary that contains loss value for
        # rotation, translation, match and combined for current step.
        self.train_loss_info: List[Dict] = []
        self.val_loss_info: List[Dict] = []
        logging.basicConfig(level=logging.INFO)

    # DONE: Calculate mean loss for each epoch
    # TODO: ADD GT TO PREDICTIONS VIS RESULTS AND APPLY SIGMOID TO MATCH LOGITS
    def fit(self, train_loader, val_loader, epochs):
        early_stopper = EarlyStopper(patience=5, min_delta=0.1)
        logging.info(
            f"""Used device: {self.device} """
        )
        # track total training time
        total_start_time = time.time()

        if self.vis_predictions_every_n:
            # take batch
            if self.vis_validation:
                vis_data = next(iter(val_loader))
            else:
                vis_data = next(iter(train_loader))
            cur_img_batch, cur_trans_img_batch, gt_is_matched, gt_transformation = vis_data
            k = 5 # take first k item from batch
            img = cur_img_batch[:k, ...]
            trans_img = cur_trans_img_batch[:k, ...]
            gt_match = gt_is_matched[:k, ...]
            gt_trans = gt_transformation[:k, ...]
            del cur_img_batch
            del cur_trans_img_batch
            del gt_is_matched
            del gt_transformation
            torch.cuda.empty_cache()
        # training
        for epoch in range(epochs):
            # track epoch time
            epoch_start_time = time.time()

            # train
            if self.vis_predictions_every_n:
                if (epoch % self.vis_predictions_every_n) == 0:
                        tx_max, ty_max = self.train_stats_config["tx_stats"]["max"], self.train_stats_config["ty_stats"]["max"]
                        # forward pass
                        if len(img.shape) != 4:
                            # add batch dim
                            img = torch.unsqueeze(img, 0)
                             
                        if len(trans_img.shape) != 4:           
                            trans_img = torch.unsqueeze(trans_img, 0)
                            
                        # get prediction from the current model state  
                        prediction = self.model(img.to(self.device), trans_img.to(self.device))
                        # get affine transformation prediction
                        prediction_affine = prediction[1]
                        prediction_match = prediction[0]
                        # normalize back the translation values 
                        prediction_affine[..., 2] = prediction_affine[..., 2] * tx_max
                        prediction_affine[..., 3] = prediction_affine[..., 3] * ty_max
                        save_prediction_results(org_img=img, trans_img=trans_img,
                                                gt_match=gt_match,
                                                gt_trans=gt_trans, 
                                                prediction_affine=prediction_affine, 
                                                prediction_match=prediction_match, 
                                                experiment_name=self.experiment_name, epoch=epoch)

            tr_loss, tr_loss_info = self._train(train_loader)

            # validate
            val_loss, val_loss_info = self._validate(val_loader)

            self.train_loss_.append(tr_loss)
            self.val_loss_.append(val_loss)

            self.train_loss_info.append(tr_loss_info)
            self.val_loss_info.append(val_loss_info)

            epoch_time = time.time() - epoch_start_time
            self._logger(
                tr_loss,
                val_loss,
                tr_loss_info,
                val_loss_info,
                epoch + 1,
                epochs,
                epoch_time,
                **self.logger_kwargs
            )
            
            if early_stopper.early_stop(validation_loss=val_loss):
                        logging.info("Early Stopping.")
                        break

        total_time = time.time() - total_start_time

        # final message
        logging.info(
            f"""End of training. Total time: {round(total_time, 5)} seconds"""
        )

    def _train(self, dataloader: DataLoader):
        self.model.train()
        mean_loss = 0.0
        loss_info = {}
        tx_max, ty_max = self.train_stats_config["tx_stats"]["max"], self.train_stats_config["ty_stats"]["max"]
        index = 0
        for cur_data in dataloader:
            # move to device
            cur_data = self._to_device(cur_data, device=self.device)
            cur_img_batch, cur_trans_img_batch, cur_gt_match_batch, cur_gt_trans_batch = cur_data
            # normalize translation values before giving it to the model
            # normalize tx (between -1 to +1)
            cur_gt_trans_batch[..., 2] = cur_gt_trans_batch[..., 2] / tx_max
            # normalize ty
            cur_gt_trans_batch[..., 3] = cur_gt_trans_batch[..., 3] / ty_max
            
            # remove gradient from previous passes
            self.optimizer.zero_grad()
            
            # forward pass
            prediction = self.model(cur_img_batch, cur_trans_img_batch)

            # loss
            loss, loss_info = self._compute_combined_loss(prediction, (cur_gt_match_batch, cur_gt_trans_batch))

            # backprop
            loss.backward()

            # parameters update
            self.optimizer.step()
            mean_loss += loss.item()
            index += 1
        mean_loss = mean_loss / index
        return mean_loss, loss_info

    def _validate(self, dataloader):
        self.model.eval()
        tx_max, ty_max = self.train_stats_config["tx_stats"]["max"], self.train_stats_config["ty_stats"]["max"]
        mean_loss = 0.0
        index = 0
        with torch.no_grad():
            for cur_data in dataloader:
                # move to device
                cur_data = self._to_device(cur_data, device=self.device)
                cur_img_batch, cur_trans_img_batch, cur_gt_match_batch, cur_gt_trans_batch = cur_data
                # normalize translation values before giving it to the model
                # normalize tx (between -1 to +1)
                cur_gt_trans_batch[..., 2] = cur_gt_trans_batch[..., 2] / tx_max
                # normalize ty
                cur_gt_trans_batch[..., 3] = cur_gt_trans_batch[..., 3] / ty_max
                # forward pass
                prediction = self.model(cur_img_batch, cur_trans_img_batch)

                # loss
                loss, loss_info = self._compute_combined_loss(prediction, (cur_gt_match_batch, cur_gt_trans_batch))
                index += 1
                mean_loss += loss.item()
        mean_loss = mean_loss / index
        return mean_loss, loss_info

    def _compute_combined_loss(self, pred: Tuple, gt: Tuple):
        # model returns matching probability and transform prediction.
        combined_loss, loss_info = self.criterion(pred, gt)
        return combined_loss, loss_info

    def _logger(self, tr_loss, val_loss, tr_loss_info, val_loss_info,
                epoch, epochs, epoch_time, show=True, update_step=20):
        if show:
            if epoch % update_step == 0 or epoch == 1:
                # to satisfy pep8 common limit of characters
                if self.show_all_losses:
                    msg = f"Epoch {epoch}/{epochs} | Train combined loss: {tr_loss} | Match Loss: {tr_loss_info['match_loss']}, Rotation Loss: {tr_loss_info['rotation_loss']} | Translation Loss {tr_loss_info['translation_loss']}"

                    msg = f"{msg} \nVal combined loss: {val_loss} | Match Loss: {val_loss_info['match_loss']}, Rotation Loss: {val_loss_info['rotation_loss']} | Translation Loss {val_loss_info['translation_loss']}"
                    msg = f"{msg} | Time/epoch: {round(epoch_time, 5)} seconds"
                    msg = msg + "\n" + "-" * 100
                    logging.info(msg)
                else:
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

    def save_model(self, path: Path, name: str):
        if not path.parent.exists():
            path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / (name + ".pt"))

    def save_experiment(self, experiments_dir: Path):
        if not experiments_dir.exists():
            experiments_dir.mkdir(parents=True, exist_ok=True)

        cur_experiment_dir = experiments_dir / self.experiment_name
        cur_experiment_dir.mkdir(parents=True, exist_ok=True)
        current_training_config = get_train_config()
        save_to_json(current_training_config, str(cur_experiment_dir / "_train_config.json"))

        # save train-val combined loss graph

        save_loss_graph(save_path=cur_experiment_dir / "combined_loss_plot.png", train_loss=self.train_loss_,
                        val_loss=self.val_loss_, titles=["Train Combined Loss", "Val Combined Loss"],
                        labels=["train loss", "val_loss"])
        # save rotation loss, match loss and translation loss
        # rotation loss
        rotation_loss_train = [i["rotation_loss"] for i in self.train_loss_info]
        rotation_loss_val = [i["rotation_loss"] for i in self.val_loss_info]
        save_loss_graph(save_path=cur_experiment_dir / "rotation_loss_plot.png", train_loss=rotation_loss_train,
                        val_loss=rotation_loss_val, titles=["Train Rotation Loss", "Val Rotation Loss"],
                        labels=["train loss", "val_loss"])
        # translation loss
        translation_loss_train = [i["translation_loss"] for i in self.train_loss_info]
        translation_loss_val = [i["translation_loss"] for i in self.val_loss_info]
        save_loss_graph(save_path=cur_experiment_dir / "translation_loss_plot.png", train_loss=translation_loss_train,
                        val_loss=translation_loss_val, titles=["Train Translation Loss", "Val Translation Loss"],
                        labels=["train loss", "val_loss"])
        # match loss
        match_loss_train = [i["match_loss"] for i in self.train_loss_info]
        match_loss_val = [i["match_loss"] for i in self.val_loss_info]
        save_loss_graph(save_path=cur_experiment_dir / "match_loss_plot.png", train_loss=match_loss_train,
                        val_loss=match_loss_val, titles=["Train Match Loss", "Val Match Loss"],
                        labels=["train loss", "val_loss"])
