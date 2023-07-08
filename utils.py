import torch
from torch import nn
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import Compose, Activations, AsDiscrete
from monai.visualize.utils import blend_images


CLASSES = ['kidney', 'tumor', 'cyst']


def get_tp_fn_fp(ground_truth, predictions):
    tp = ((ground_truth == predictions) & (ground_truth == 1))
    fn = ((ground_truth != predictions) & (ground_truth == 1))
    fp = ((ground_truth != predictions) & (ground_truth == 0))

    return tp, fn, fp

def blend_imgs(img, label, pred):
    tp, fn, fp = get_tp_fn_fp(label, pred)

    out = blend_images(img, tp, alpha=0.5, cmap='Greens')
    out = blend_images(out, fn, alpha=0.5, cmap='Reds')
    out = blend_images(out, fp, alpha=0.5, cmap='summer')

    return out

def postprocess(output, threshold):
    # postprocessing = Compose([Activations(softmax=True, dim=1),
    #                           AsDiscrete(threshold=threshold)])
    postprocessing = AsDiscrete(argmax=True, to_onehot=4, dim=1)

    return postprocessing(output)

def compute_metrics(y_true, y_pred):
    one_hot = AsDiscrete(to_onehot=4, dim=1)
    y_true = one_hot(y_true)

    dice_metric = DiceMetric(include_background=False, ignore_empty=False)
    iou_metric = MeanIoU(include_background=False, ignore_empty=False)

    dice = dice_metric(y_pred, y_true)
    iou = iou_metric(y_pred, y_true)

    dice = dice_metric.aggregate(reduction='mean_batch')
    iou = iou_metric.aggregate(reduction='mean_batch')

    metrics_dict = dict()
    for i, cls in enumerate(CLASSES):
        metrics_dict[f'_Dice/{cls}'] = dice[i]
        metrics_dict[f'_IoU/{cls}'] = iou[i]

    metrics_dict.update({'/Dice': dice_metric.aggregate(reduction='mean'), '/IoU': iou_metric.aggregate(reduction='mean')})

    return metrics_dict