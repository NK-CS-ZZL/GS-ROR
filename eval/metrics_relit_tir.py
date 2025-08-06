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


def evaluate(gt_path, image_path):
    image_path = image_path[0]
    gt_path = gt_path[0]
    full_dict = {}
    per_view_dict = {}
    print("")
    for scene_dir in sorted(os.listdir(image_path)):
        print("Env light:", scene_dir)
        full_dict[scene_dir] = {}
        test_dir = Path(image_path) / scene_dir / "test/ours_24000/renders"

        ssims = []
        psnrs = []
        lpipss = []
        img_names = []
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        pbar = tqdm(enumerate(sorted(os.listdir(test_dir))), total=len(os.listdir(test_dir)))
        for idx, method in pbar:
            per_view_dict[scene_dir][method] = {}

            
            renders_dir = test_dir / method
            gt_dir = Path(gt_path) / f"test_{idx:03d}"
            render = imageio.imread(renders_dir)
            gt = imageio.imread(gt_dir / f'rgba_{scene_dir}.png')
            gt = tf.to_tensor(gt).unsqueeze(0).cuda()
            mask = gt[:, -1, :, :]
            gt = gt[:, :3, :, :] * mask + 1 - mask
            render = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()

            ssims.append(ssim(render, gt).squeeze())
            psnrs.append(psnr(render, gt).squeeze())
            lpipss.append(lpips(render, gt, net_type='vgg').squeeze())
            img_names.append(f"test_{idx:03d}")
            pbar.set_description_str(f"[{scene_dir}] psnr: {psnrs[-1].item():.2f}; ssim: {ssims[-1]:.4f}; lpips: {lpipss[-1]:.4f}")

            per_view_dict[scene_dir][method].update({"SSIM": {img_names[-1]: ssims[-1].item()},
                                                        "PSNR": {img_names[-1]: psnrs[-1].item()},
                                                        "LPIPS": {img_names[-1]: lpipss[-1].item()}})
        full_dict[scene_dir].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        with open(image_path + f"/{scene_dir}/results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(image_path + f"/{scene_dir}/per_view.json", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)
        # except:
        #     print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--gt_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--img_paths', '-i', required=True, nargs="+", type=str, default=[])
    
    args = parser.parse_args()
    evaluate(args.gt_paths, args.img_paths)
