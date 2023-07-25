import argparse
import importlib
from glob import glob
from collections import OrderedDict, defaultdict
from tqdm import tqdm
import time
import numpy as np

import torch
import pytorch_lightning as pl

from monai.inferers import sliding_window_inference
from monai.transforms import SaveImage
from monai.data import CacheDataset, DataLoader

from monai.transforms import Compose, LoadImaged, Orientationd, Spacingd, Lambdad, NormalizeIntensityd, CropForegroundd
from monai.transforms.utils import allow_missing_keys_mode
from torch import clamp

from utils import *

DATASET_DIR='/kits23-dataset'


# class KITSValDataModule(pl.LightningDataModule):
#     def __init__(self, params, val_transforms):
#         super().__init__()

#         self.params = params
#         self.val_transforms = val_transforms

#     def setup(self, stage=None):
#         if stage == "fit" or stage is None:
#             all_images = sorted(glob(DATASET_DIR+'/*/imaging.nii.gz'))
#             all_labels = sorted(glob(DATASET_DIR+'/*/segmentation.nii.gz'))

#             val_images = all_images[int(0.8 * len(all_images)) :]#[:2]
#             val_labels = all_labels[int(0.8 * len(all_labels)) :]#[:2]
#             val_files = [{"image": img, "label": lbl, "case": img.split('/')[-2]} for img, lbl in zip(val_images, val_labels)]
#             print("Number of validation files:", len(val_files))
#             self.val_dataset = CacheDataset(val_files, transform=self.val_transforms, cache_rate=0.1)

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             **self.params["val_dataloader"],
#         )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("params", type=str, help="Path to parameters py file")
    parser.add_argument("ckpt", type=str, help="Path to saved lightning checkpoint")
    parser.add_argument("--save_num_imgs", default=0, type=int, help="Save this many predictions")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("params", args.params)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)

    val_transforms = Compose([
            LoadImaged(
                keys=["image", "label"],
                image_only=False,
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
            CropForegroundd(
                keys=["image", "label"],
                source_key='image')
          ])

    all_images = sorted(glob(DATASET_DIR+'/*/imaging.nii.gz'))
    all_labels = sorted(glob(DATASET_DIR+'/*/segmentation.nii.gz'))

    val_images = all_images[int(0.8 * len(all_images)) :]
    val_labels = all_labels[int(0.8 * len(all_labels)) :]
    val_files = [{"image": img, "label": lbl, "case": img.split('/')[-2]} for img, lbl in zip(val_images, val_labels)]
    print("Number of validation files:", len(val_files))
    
    # data_module = KITSValDataModule(
    #     params.PARAMS, val_transforms
    # )
    # data_module.setup()
    # val_dataloader = data_module.val_dataloader() 

    model = params.model
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    checkpoint = OrderedDict({k.replace('model.', '', 1):v for k,v in checkpoint['state_dict'].items()})
    model.load_state_dict(checkpoint)
    model.half()
    model.to(torch.device('cuda'))
    model.eval()

    all_metrics = defaultdict(float)
    argmax = AsDiscrete(argmax=True, dim=1)
    
    times = []
    with torch.no_grad():
        # for i, batch in enumerate(tqdm(val_dataloader)):
        for i, path_dict in tqdm(enumerate(val_files)):
            time0 = time.time()
            # image = batch["image"].as_tensor().half().to(torch.device('cuda'))
            # label = batch["label"].as_tensor()
            inputs = val_transforms(path_dict)
            image = inputs['image'].unsqueeze(0).half().to(torch.device('cuda'))
            label = inputs['label'].unsqueeze(0)

            outputs = sliding_window_inference(
                        inputs=image,
                        roi_size=params.PARAMS["patch_size"],
                        sw_batch_size=1,
                        predictor=model,
                        overlap=params.PARAMS["overlap"],
                        mode="gaussian",
                    )

            post_outputs = postprocess(outputs.to(torch.float).cpu(), params.PARAMS['threshold'])
            times.append(time.time()-time0)
            metrics = compute_metrics(label, post_outputs)
            for k in metrics.keys():
                all_metrics[k] += metrics[k].item()

            # inverse transforms
            post_outputs = argmax(post_outputs)
            post_outputs.applied_operations = inputs['label'].applied_operations
            post_outputs.meta = inputs['label'].meta

            post_outputs = val_transforms.inverse({'image': inputs['image'], 'label': post_outputs[0]})['label']
            # post_input = val_transforms.inverse({'image': inputs['image'], 'label': inputs['label']})
            
            # save images
            model_name = args.ckpt.split('/')[-4] + '_' + args.ckpt.split('/')[-3]
            # case = batch["case"][0]
            case = path_dict["case"]
            if i < args.save_num_imgs:
                # # for img, name in zip([image.cpu(), label, post_outputs['label']], ['ct', 'true', 'pred']):
                # for img, name in zip([post_input['image'], post_input['label'], post_outputs], ['ct', 'true', 'pred']):
                #     save = SaveImage(output_dir=f'./results/{model_name}/{case}/', output_postfix=name)
                #     save(img[0])
                save = SaveImage(output_dir=f'./results/{model_name}/{case}', output_postfix='')
                save(post_outputs[0], inputs['label'].meta)

    # print metrics
    for k, v in all_metrics.items():
        # all_metrics[k] = round(v/len(val_dataloader),3)
        all_metrics[k] = round(v/len(val_files),3)
    print('Dice', all_metrics['/Dice'], '\nKidney', all_metrics['_Dice/kidney'], '\nTumor', all_metrics['_Dice/tumor'], '\nCyst', all_metrics['_Dice/cyst'])
    print('\nIoU', all_metrics['/IoU'], '\nKidney', all_metrics['_IoU/kidney'], '\nTumor', all_metrics['_IoU/tumor'], '\nCyst', all_metrics['_IoU/cyst'])
    print('\nMean Time', round(np.mean(times), 3), 's')