#add test file for dataset
import argparse
import importlib
from glob import glob
from collections import OrderedDict
from tqdm import tqdm

import torch

from monai.inferers import sliding_window_inference
from monai.transforms import SaveImage

from monai.transforms import Compose, LoadImaged, Orientationd, Spacingd, Lambdad, NormalizeIntensityd, CropForegroundd
from torch import clamp

from utils import *

TEST_DIR = '230720_kits23_test_images'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("params", type=str, help="Path to parameters py file")
    parser.add_argument("ckpt", type=str, help="Path to saved lightning checkpoint")
    #parser.add_argument("--output_dir", type=str, default='./results/test/', help="Path to save the predictions")
    args = parser.parse_args()
    #output_dir = args.output_dir

    output_dir = f"./results/test_{args.ckpt.split('/')[-4]}_{args.ckpt.split('/')[-3]}/"
    print(output_dir)
    spec = importlib.util.spec_from_file_location("params", args.params)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)

    test_images = glob(TEST_DIR + '/*.nii.gz')

    model = params.model
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    checkpoint = OrderedDict({k.replace('model.', '', 1):v for k,v in checkpoint['state_dict'].items()})
    model.load_state_dict(checkpoint)
    model.half()
    model.to(torch.device('cuda'))
    model.eval()

    argmax = AsDiscrete(argmax=True, dim=1)

    preprocess = Compose([
            LoadImaged(
                keys=["image"],
                image_only=False,
                ensure_channel_first=True,
            ),
            Orientationd(
                keys=["image"],
                axcodes="RAS",
            ),
            Spacingd(
                keys=["image"],
                pixdim=(0.78125, 0.78125, 3.0),
                mode="bilinear",
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
                keys="image",
                source_key='image')
          ])

    with torch.no_grad():
        for img_path in tqdm(test_images):
            image = preprocess({'image': img_path})
            input_img = image['image'].unsqueeze(0).half().to(torch.device('cuda'))

            outputs = sliding_window_inference(
                        inputs=input_img,
                        roi_size=params.PARAMS["patch_size"],
                        sw_batch_size=1,
                        predictor=model,
                        overlap=params.PARAMS["overlap"],
                        mode="gaussian",
                    )

            post_outputs = postprocess(outputs.to(torch.float).cpu(), params.PARAMS['threshold'])
            post_outputs = argmax(post_outputs)

            post_outputs.applied_operations = image['image'].applied_operations
            post_outputs.meta = image['image'].meta
            post_outputs = preprocess.inverse({"image": post_outputs[0]})

            # save images
            case = image['image_meta_dict']['filename_or_obj'].split('/')[-1].split('.')[0]
            save = SaveImage(output_dir=output_dir, output_postfix='')
            save(post_outputs['image'][0], image['image'].meta)
