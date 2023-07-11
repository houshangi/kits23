from __future__ import annotations
import numpy as np
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
import torch
import torch.optim
from torch import clamp
from monai.utils import DiceCEReduction, LossReduction, Weight, look_up_option, pytorch_after
from torch.nn.modules.loss import _Loss
from collections.abc import Callable, Sequence

from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    # ScaleIntensityd,
    RandCropByLabelClassesd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    SpatialPadd,
    Spacingd,
    Orientationd,
    Lambdad,
    CropForegroundd
)


class DiceCELossLabelSmoothing(_Loss):
    """
    Compute both Dice loss and Cross Entropy Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Cross Entropy Loss is shown in ``torch.nn.CrossEntropyLoss``. In this implementation,
    two deprecated parameters ``size_average`` and ``reduce``, and the parameter ``ignore_index`` are
    not supported.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: torch.Tensor | None = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> None:
        """
        Args:
            ``ce_weight`` and ``lambda_ce`` are only used for cross entropy loss.
            ``reduction`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``. only used by the `DiceLoss`, not for the `CrossEntropyLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            ce_weight: a rescaling weight given to each class for cross entropy loss.
                See ``torch.nn.CrossEntropyLoss()`` for more information.
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_ce: the trade-off weight value for cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        reduction = look_up_option(reduction, DiceCEReduction).value
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=ce_weight, reduction=reduction, label_smoothing=0.1)
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.old_pt_ver = not pytorch_after(1, 10)


    def ce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute CrossEntropy loss for the input and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif self.old_pt_ver:
            warnings.warn(
                f"Multichannel targets are not supported in this older Pytorch version {torch.__version__}. "
                "Using argmax (as a workaround) to convert target to a single channel."
            )
            target = torch.argmax(target, dim=1)
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.cross_entropy(input, target)  # type: ignore[no-any-return]


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )

        dice_loss = self.dice(input, target)
        ce_loss = self.ce(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        return total_loss



PARAMS = dict(
    epochs=100,
    exp_name="swin-256-label-smoothing",
    acc_batch_size=4,
    patch_size=(256, 256, 32),
    num_samples=2,
    optimizer=torch.optim.RAdam,
    optimizer_params=dict(lr=1e-4, weight_decay=0),
    scheduler=torch.optim.lr_scheduler.OneCycleLR,
    scheduler_params=dict(
        max_lr=3e-4, div_factor=10, final_div_factor=100, pct_start=0.1
    ),
    accelerator="gpu",
    devices=2,
    precision=16,
    seed=2023,
    ckpt_monitor="val/Dice",
    log_every_n_batch=2,
    log_val_imgs=2,
    train_dataloader=dict(
        batch_size=2,
        shuffle=True,
        num_workers=10,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    ),
    val_dataloader=dict(
        batch_size=1,
        num_workers=5,
        pin_memory=True,
        persistent_workers=True,
    ),
    # validation
    sw_batch_size=2,
    threshold=0.5,
    overlap=0.5
)


model_params = dict(
    img_size=PARAMS['patch_size'],
    in_channels=1,
    out_channels=4,
    feature_size=24
)

model = SwinUNETR(**model_params)


def loss_function(outputs, labels):
    loss_fn = DiceCELossLabelSmoothing(to_onehot_y=True, softmax=True, include_background=False)
    loss = loss_fn(outputs, labels)

    return loss


def get_preprocess_transforms():
    return [
            LoadImaged(
                keys=["image", "label"],
                image_only=True,
                ensure_channel_first=True,
            ),
            Orientationd(
                keys=["image", "label"],
                axcodes="RAS",
            ),
            Spacingd(
                keys=["image", "label"],
                pixdim=(0.78125, 0.78125, 3.0),
                mode=("bilinear", "nearest"),
            ),
            Lambdad(
                keys="image",
                func=lambda x: clamp(x, -200, 300),
            ),
            NormalizeIntensityd(
                keys="image",
                nonzero=True,
            ),
            # ScaleIntensityd(keys="image"),
          ]

def get_val_transforms():
    preprocess = get_preprocess_transforms()
    preprocess.append(CropForegroundd(keys=["image", "label"],
                                      source_key="image"))
    return Compose(preprocess)


def get_train_transforms():
    preprocess = get_preprocess_transforms()

    augmentations = [
            SpatialPadd(
                keys=("image", "label"),
                spatial_size=PARAMS["patch_size"],
            ),
            RandCropByLabelClassesd(
                keys=("image", "label"),
                label_key="label",
                spatial_size=PARAMS["patch_size"],
                num_samples=PARAMS["num_samples"],
                allow_smaller=True,
                ratios=[1, 5, 10, 20],
                num_classes=4,
            ),
            RandAffined(
                keys=("image", "label"),
                prob=0.75,
                rotate_range=(np.pi / 6, np.pi / 6),
                translate_range=(0.0625, 0.0625),
                scale_range=(0.1, 0.1),
            ),
            RandFlipd(
                keys=("image", "label"),
                spatial_axis=(0, 1, 2),
                prob=0.5,
            ),
            RandGaussianNoised(
                keys="image",
                prob=0.15,
                mean=0.0,
                std=0.1,
            ),
            RandGaussianSmoothd(
                keys="image",
                prob=0.15,
                sigma_x=(0.5, 1.15),
                sigma_y=(0.5, 1.15),
            ),
            RandScaleIntensityd(
                keys="image",
                factors=0.3,
                prob=0.1,
            ),
        ]

    return Compose(
        preprocess + augmentations
    )
