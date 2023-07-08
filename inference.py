import argparse
import importlib
from glob import glob
from collections import OrderedDict, defaultdict
from tqdm import tqdm

import torch
import pytorch_lightning as pl

from monai.inferers import sliding_window_inference
from monai.transforms import SaveImage
from monai.data import CacheDataset, DataLoader

from utils import *

DATASET_DIR='/kits23-dataset'


class KITSValDataModule(pl.LightningDataModule):
    def __init__(self, params, val_transforms):
        super().__init__()

        self.params = params
        self.val_transforms = val_transforms

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            all_images = sorted(glob(DATASET_DIR+'/*/imaging.nii.gz'))
            all_labels = sorted(glob(DATASET_DIR+'/*/segmentation.nii.gz'))

            val_images = all_images[int(0.8 * len(all_images)) :][:2]
            val_labels = all_labels[int(0.8 * len(all_labels)) :][:2]
            val_files = [{"image": img, "label": lbl, "case": img.split('/')[-2]} for img, lbl in zip(val_images, val_labels)]
            print("Number of validation files:", len(val_files))
            self.val_dataset = CacheDataset(val_files, transform=self.val_transforms, cache_rate=0.1)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            **self.params["val_dataloader"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get all command line arguments.")
    parser.add_argument("params", type=str, help="Path to parameters py file")
    parser.add_argument("ckpt", type=str, help="Path to saved lightning checkpoint")
    parser.add_argument("--save_num_imgs", default=0, type=int, help="Save this many predictions")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("params", args.params)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)

    data_module = KITSValDataModule(
        params.PARAMS, params.get_val_transforms()
    )
    data_module.setup()
    val_dataloader = data_module.val_dataloader() 

    model = params.model
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    checkpoint = OrderedDict({k.replace('model.', '', 1):v for k,v in checkpoint['state_dict'].items()})
    model.load_state_dict(checkpoint)
    model.half()
    model.to(torch.device('cuda'))
    model.eval()

    all_metrics = defaultdict(float)
    argmax = AsDiscrete(argmax=True, dim=1)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader)):
            image = batch["image"].as_tensor().half().to(torch.device('cuda'))
            label = batch["label"].as_tensor()

            outputs = sliding_window_inference(
                        inputs=image,
                        roi_size=params.PARAMS["patch_size"],
                        sw_batch_size=1,
                        predictor=model,
                        overlap=params.PARAMS["overlap"],
                        mode="gaussian",
                    )

            post_outputs = postprocess(outputs.to(torch.float).cpu(), params.PARAMS['threshold'])
            metrics = compute_metrics(label, post_outputs)
            for k in metrics.keys():
                all_metrics[k] += metrics[k].item()

            # save images
            model_name = args.ckpt.split('/')[-4] + '_' + args.ckpt.split('/')[-3]
            case = batch["case"][0]
            if i < args.save_num_imgs:
                for img, name in zip([image.cpu(), label, argmax(post_outputs)], ['ct', 'true', 'pred']):
                    save = SaveImage(output_dir=f'./results/{model_name}/{case}/', output_postfix=name)
                    save(img[0])

    # print metrics
    for k, v in all_metrics.items():
        all_metrics[k] = round(v/len(val_dataloader),3)
    print('Dice', all_metrics['/Dice'], '\nKidney', all_metrics['_Dice/kidney'], '\nTumor', all_metrics['_Dice/tumor'], '\nCyst', all_metrics['_Dice/cyst'])
    print('\nIoU', all_metrics['/IoU'], '\nKidney', all_metrics['_IoU/kidney'], '\nTumor', all_metrics['_IoU/tumor'], '\nCyst', all_metrics['_IoU/cyst'])
    