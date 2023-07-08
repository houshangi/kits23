import numpy as np
from pathlib import Path
# from monai.networks.nets import UNet
from monai.losses import DiceCELoss, DiceLoss
from torch.nn import CrossEntropyLoss
import torch.optim
import torch
from x_unet import XUnet

from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    # ScaleIntensityd,
    RandCropByPosNegLabeld,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandZoomd,
    SpatialPadd,
    Spacingd,
    Orientationd
)


PARAMS = dict(
    epochs=100,
    exp_name="u2net",
    acc_batch_size=4,
    patch_size=(64, 64, 64),
    num_samples=6,
    optimizer=torch.optim.RAdam,
    optimizer_params=dict(lr=1e-3, weight_decay=0),
    scheduler=torch.optim.lr_scheduler.OneCycleLR,
    scheduler_params=dict(
        max_lr=3e-4, div_factor=10, final_div_factor=100, pct_start=0.1
    ),
    accelerator="gpu",
    devices=2,
    precision=16,
    seed=2023,
    ckpt_monitor="val/dice",
    log_every_n_batch=2,
    log_val_imgs=2,
    train_dataloader=dict(
        batch_size=4,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=1,
        persistent_workers=True,
    ),
    val_dataloader=dict(
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    ),
    # validation
    sw_batch_size=4,
    threshold=0.5,
)


model_params = dict(dim = 16,
                    frame_kernel_size = 3,
                    channels = 1,
                    out_dim = 4,
                    # attn_dim_head = 8,
                    # attn_heads = 8,
                    dim_mults = (1, 2, 4),#, 8),
                    num_blocks_per_stage = (2, 2, 2),#, 2),
                    num_self_attn_per_stage = (0, 0, 0),#, 0),
                    nested_unet_depths = (4, 2, 1),#(5, 4, 2, 1),
                    consolidate_upsample_fmaps = False,
                    weight_standardize = False
)

model = XUnet(**model_params)


def loss_function(outputs, labels):
    # loss_fn = CrossEntropyLoss()
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    loss = loss_fn(outputs, labels)

    return loss


def get_train_transforms():
    return Compose(
        [
            LoadImaged(
                keys=["image", "label"],
                image_only=True,
                ensure_channel_first=True,
            ),
            # AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True),
            # ScaleIntensityd(keys="image"),
            SpatialPadd(keys=("image", "label"), spatial_size=PARAMS["patch_size"]),
            RandCropByPosNegLabeld(
                keys=("image", "label"),
                label_key="label",
                spatial_size=PARAMS["patch_size"],
                num_samples=PARAMS["num_samples"],
                image_key="image",
                image_threshold=0,
                allow_smaller=True,
            ),
            RandAffined(
                keys=("image", "label"),
                prob=0.75,
                rotate_range=(np.pi / 4, np.pi / 4),
                translate_range=(0.0625, 0.0625),
                scale_range=(0.1, 0.1),
            ),
            RandFlipd(
                keys=("image", "label"), spatial_axis=0, prob=0.5
            ),
            RandFlipd(
                keys=("image", "label"), spatial_axis=1, prob=0.5
            ),
            RandGaussianNoised(keys="image", prob=0.15, mean=0.0, std=0.01),
            RandGaussianSmoothd(
                keys="image", prob=0.15, sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15)
            ),
            RandScaleIntensityd(keys="image", factors=0.3, prob=0.15),
            # RandZoomd(
            #     keys=("image", "label"),
            #     min_zoom=0.9,
            #     max_zoom=1.2,
            #     mode=("bilinear", "nearest"),
            #     align_corners=(True, None),
            #     prob=0.15,
            # ),
        ]
    )


def get_val_transforms():
    return Compose(
        [
            LoadImaged(
                keys=["image", "label"],
                image_only=True,
                ensure_channel_first=True,
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            # AddChanneld(keys=["image", "label"]),
            NormalizeIntensityd(keys="image", nonzero=True),
            # ScaleIntensityd(keys="image"),
        ]
    )
