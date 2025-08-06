#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import imageio
import torch.nn.functional as F


def compute_mae(normals, normals_gt, masks):
    one_eps = torch.tensor(1 - torch.finfo(torch.float32).eps).cuda()
    return (masks * torch.arccos(torch.clip((normals * normals_gt).sum(-1), -one_eps, one_eps))).sum() / masks.sum() * 180.0 / torch.pi


def evaluate(gt_path, image_path):
    scene = os.path.split(image_path[0].rstrip('/'))[-1]
    image_path = image_path[0]
    gt_path = gt_path[0]
    full_dict = {}
    per_view_dict = {}

    full_dict = {}
    test_dir = Path(image_path) / "test" / "ours_24000" / "normal"

    maes = []
    img_names = []
    full_dict = {}
    per_view_dict = {}
    pbar = tqdm(enumerate(sorted(os.listdir(test_dir))), total=len(os.listdir(test_dir)))
    for idx, method in pbar:
        per_view_dict[method] = {}

        renders_dir = test_dir / method
        gt_dir = Path(gt_path) 
        print(renders_dir)
        print(gt_dir / f'r_{idx}_normal.png')
        render = imageio.imread(renders_dir)
        gt = imageio.imread(gt_dir / f'r_{idx}_normal.png')
        if os.path.exists(gt_dir / f'r_{idx}_alpha.png'):
            img = imageio.imread(gt_dir / f'r_{idx}_alpha.png')
            img = tf.to_tensor(img).unsqueeze(0).cuda()
            mask = img[:, :1, :, :].squeeze()
        else:
            img = imageio.imread(gt_dir / f'r_{idx}.png')
            img = tf.to_tensor(img).unsqueeze(0).cuda()
            mask = img[:, 3:, :, :].squeeze()
        gt = tf.to_tensor(gt).unsqueeze(0).cuda()
        render = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        gt = gt[:, :3, :, :] * 2 - 1
        gt = F.normalize(gt, p=2, dim=1).squeeze(0).permute(1, 2, 0)
        render = render * 2 - 1
        render = F.normalize(render, p=2, dim=1).squeeze(0).permute(1, 2, 0)


        maes.append(compute_mae(render, gt, mask).squeeze())
        img_names.append(f"test_{idx:03d}")
        pbar.set_description_str(f" mae: {maes[-1].item():.2f}")


        per_view_dict[method].update({
                                        "MAE": {img_names[-1]: maes[-1].item()},
                                    })
        full_dict.update({
                            "MAE": torch.tensor(maes).mean().item(),
                        })
        print("  MAE : {:>12.7f}".format(torch.tensor(maes).mean(), ".5"))
        print("")

        with open(image_path + f"/mae_results.json", 'w') as fp:
            json.dump(full_dict, fp, indent=True)
        with open(image_path + f"//mae_per_view.json", 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--gt_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--img_paths', '-i', required=True, nargs="+", type=str, default=[])
    
    args = parser.parse_args()
    evaluate(args.gt_paths, args.img_paths)
