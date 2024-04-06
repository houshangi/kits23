#!/usr/bin/env python
# coding: utf-8

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import argparse
import importlib
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from  monai.utils import set_determinism
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference


from utils import *

DATASET_DIR='/kits23-dataset'

class KITSDataModule(pl.LightningDataModule):
    def __init__(self, params, train_transforms, val_transforms):
        super().__init__()

        self.params = params
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            all_images = sorted(glob(DATASET_DIR+'/*/imaging.nii.gz'))#[:10]
            all_labels = sorted(glob(DATASET_DIR+'/*/segmentation.nii.gz'))#[:10]

            train_images = all_images[: int(0.8 * len(all_images))]
            train_labels = all_labels[: int(0.8 * len(all_labels))]
            train_files = [{"image": img, "label": lbl} for img, lbl in zip(train_images, train_labels)]
            print("Number of training files:", len(train_files))
            self.train_dataset = CacheDataset(train_files, transform=self.train_transforms, cache_rate=0.1)

            val_images = all_images[int(0.8 * len(all_images)) :]
            val_labels = all_labels[int(0.8 * len(all_labels)) :]
            val_files = [{"image": img, "label": lbl} for img, lbl in zip(val_images, val_labels)]
            print("Number of validation files:", len(val_files))
            self.val_dataset = CacheDataset(val_files, transform=self.val_transforms, cache_rate=0.1)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            **self.params["train_dataloader"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            **self.params["val_dataloader"],
        )



class KITSModule(pl.LightningModule):
    def __init__(
        self,
        params,
        model,
        loss_function,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model", "loss_function"])

        self.params = params
        self.model = model
        self.loss = loss_function

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def configure_optimizers(self):
        optimizer = self.params["optimizer"](
            self.model.parameters(), **self.params["optimizer_params"]
        )

        if self.params["scheduler"]:
            if self.params["scheduler"].__name__ == "OneCycleLR":
                scheduler = self.params["scheduler"](
                    optimizer,
                    total_steps=self.trainer.estimated_stepping_batches,
                    **self.params["scheduler_params"],
                )
                scheduler = {"scheduler": scheduler, "interval": "step"}

            else:
                scheduler = self.params["scheduler"](
                    optimizer, **self.params["scheduler_params"]
                )

            optimizer_dict = {"optimizer": optimizer, "lr_scheduler": scheduler}

            return optimizer_dict

        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        time0 = time.time()
        volumes = batch["image"].as_tensor()
        labels = batch["label"].as_tensor()

        outputs = self(volumes)

        loss = self.loss(outputs, labels)
        self.log('train/loss', loss, batch_size=len(outputs))

        post_outputs = postprocess(outputs.detach().to(torch.float), self.params['threshold'])
        metrics = compute_metrics(labels, post_outputs)
        # print('METRICS', metrics)
        self.training_step_outputs.append(metrics)
        # print('LIST', self.training_step_outputs)

        if batch_idx % self.params['log_every_n_batch'] == 0:
            try:
                nonzero_batch = torch.nonzero(labels, as_tuple=True)[0][0]
                if nonzero_batch:
                    self._log_image(volumes[nonzero_batch].cpu(), labels[nonzero_batch].cpu(), post_outputs[nonzero_batch].cpu(), "train")
            except Exception as e:
                print(e)
                import pdb
                pdb.set_trace()

        self.log('train/step_time', time.time()-time0)

        return loss
    
    def on_train_epoch_end(self):
        self._log_metrics(self.training_step_outputs, 'train')
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        time0 = time.time()
        volumes = batch["image"].as_tensor()
        labels = batch["label"].as_tensor()

        outputs = sliding_window_inference(
                      inputs=volumes,
                      roi_size=self.params["patch_size"],
                      sw_batch_size=self.params["sw_batch_size"],
                      predictor=self,
                      overlap=self.params["overlap"],
                      mode="gaussian",
                  )

        loss = self.loss(outputs, labels)
        self.log('val/loss', loss, batch_size=len(outputs))

        post_outputs = postprocess(outputs.detach().to(torch.float), self.params['threshold'])
        metrics = compute_metrics(labels, post_outputs)
        self.validation_step_outputs.append(metrics)
        
        if batch_idx < self.params["log_val_imgs"]:
            self._log_image(volumes[0].cpu(), labels[0].cpu(), post_outputs[0].cpu(), f"val/{batch_idx}")

        self.log('val/step_time', time.time()-time0, on_step=True)
    
    def on_validation_epoch_end(self):
        self._log_metrics(self.validation_step_outputs, 'val')
        self.validation_step_outputs.clear()
    
    def _log_metrics(self, metrics_list, split):
        for k in metrics_list[0].keys():
            total_metric = np.mean([x[k].cpu().item() for x in metrics_list])
            self.log(f'{split}{k}', total_metric, on_epoch=True, sync_dist=False)
        
    def _log_image(self, image, label, output, stage):
        fig = plt.figure(figsize=(20,7))

        for i, cls in enumerate(CLASSES, start=1):
            lbl = torch.where(label==i, 1, 0)
            # if lbl.max() == 0:
            #     continue
            blended = blend_imgs(image, lbl, output[i].unsqueeze(0))
            blended = np.transpose(blended.numpy(), (1,2,3,0))
            # nonzero = np.nonzero(lbl.numpy())[-1]
            # slice_ = int(np.mean((nonzero.min(), nonzero.max())))
            slice_ = np.argmax(np.sum(lbl.numpy(), axis=(0,1,2)))

            plt.subplot(1,3,i)
            plt.title(cls)
            plt.imshow(blended[:,:,slice_])
            plt.axis("off")

        plt.tight_layout()
        self.logger.experiment.add_figure(f"{stage}", fig, self.current_epoch)
        plt.close()


def train(params_path, params):
    set_determinism(params.PARAMS["seed"])
    pl.seed_everything(params.PARAMS["seed"], workers=True)
    
    print("Creating data module")
    data_module = KITSDataModule(
        params.PARAMS, params.get_train_transforms(), params.get_val_transforms()
    )

    module = KITSModule(params.PARAMS, params.model, params.loss_function)

    tb_logger = TensorBoardLogger(
        save_dir="./logs", name=params.PARAMS["exp_name"], default_hp_metric=False
    )

    model_checkpoint_dir = os.path.join(
        "./logs",
        params.PARAMS["exp_name"],
        "version_" + str(tb_logger.version),
        "models",
    )
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        save_top_k=3,
        monitor=params.PARAMS["ckpt_monitor"],
        mode="max",
        dirpath=model_checkpoint_dir,
        filename="epoch={epoch:02d}-{step}-dice={val/Dice:.5f}",
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        accelerator=params.PARAMS["accelerator"],
        benchmark=True,
        check_val_every_n_epoch=10,
        devices=1,
        callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint_callback],
        logger=tb_logger,
        log_every_n_steps=1,
        max_epochs=params.PARAMS["epochs"],
        accumulate_grad_batches=max(
            1, params.PARAMS["acc_batch_size"] // params.PARAMS["train_dataloader"]["batch_size"]
        ),
        precision=params.PARAMS["precision"],
        strategy="ddp_find_unused_parameters_false" if params.PARAMS["devices"] > 1 else None,
    )

    with open(params_path) as f:
        write_params = f.read()
    trainer.logger.experiment.add_text(params_path, write_params)

    print("FIT!")

    trainer.fit(module, datamodule=data_module, ckpt_path=params.PARAMS.get("ckpt_path"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get all command line arguments.")
    parser.add_argument("params", type=str, help="Path to parameters py file")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("params", args.params)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)

    train(args.params, params)
