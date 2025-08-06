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
    scene = os.path.split(image_path[0].rstrip('/'))[-1]
    image_path = image_path[0] + '/relight'
    gt_path = gt_path[0]
    full_dict = {}
    per_view_dict = {}
    print("")

    for env_light in os.listdir(image_path):
        if os.path.isdir(Path(image_path) / env_light) is False:
            continue
        print("Env light:", env_light)
        full_dict[env_light] = {}
        test_dir = Path(image_path) / env_light / "test" / "ours_24000" / "renders"

        ssims = []
        psnrs = []
        lpipss = []
        scaled_ssims = []
        scaled_psnrs = []
        scaled_lpipss = []
        img_names = []
        full_dict[env_light] = {}
        per_view_dict[env_light] = {}
        pbar = tqdm(enumerate(sorted(os.listdir(test_dir))), total=len(os.listdir(test_dir)))
        for idx, method in pbar:
            if idx >= 16:
                break
            per_view_dict[env_light][method] = {}

            renders_dir = test_dir / method
            print(renders_dir)
            gt_dir = Path(gt_path + f'_{env_light}') 
            print(gt_dir / f'{idx}.png')
            render = imageio.imread(renders_dir)
            gt = imageio.imread(gt_dir / f'{idx}.png')
            gt = tf.to_tensor(gt).unsqueeze(0).cuda()
            mask = gt[:, 3:, :, :]
            gt = gt[:, :3, :, :] * mask + 1 - mask
            render = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda() # 1, 3, H, W
            scale = (render * gt).view(3, -1).sum(-1) / (render * render).view(3, -1).sum(-1)
            print('scale facter: ',  scale)
            scaled_render = scale[None, :, None, None] * render

            ssims.append(ssim(render, gt).squeeze())
            psnrs.append(psnr(render, gt).squeeze())
            lpipss.append(lpips(render, gt, net_type='vgg').squeeze())
            scaled_ssims.append(ssim(scaled_render, gt).squeeze())
            scaled_psnrs.append(psnr(scaled_render, gt).squeeze())
            scaled_lpipss.append(lpips(scaled_render, gt, net_type='vgg').squeeze())
            img_names.append(f"test_{idx:03d}")
            pbar.set_description_str(f"[{env_light}] psnr: {psnrs[-1].item():.2f}; ssim: {ssims[-1]:.4f}; lpips: {lpipss[-1]:.4f}")


            per_view_dict[env_light][method].update({"SSIM": {img_names[-1]: ssims[-1].item()},
                                                        "PSNR": {img_names[-1]: psnrs[-1].item()},
                                                        "LPIPS": {img_names[-1]: lpipss[-1].item()},
                                                        "SCALED_SSIM": {img_names[-1]: scaled_ssims[-1].item()},
                                                        "SCALED_PSNR": {img_names[-1]: scaled_psnrs[-1].item()},
                                                        "SCALED_LPIPS": {img_names[-1]: scaled_lpipss[-1].item()},
                                                        })
        full_dict[env_light].update({"SSIM": torch.tensor(ssims).mean().item(),
                                     "PSNR": torch.tensor(psnrs).mean().item(),
                                     "LPIPS": torch.tensor(lpipss).mean().item(),
                                     "SCALED_SSIM": torch.tensor(scaled_ssims).mean().item(),
                                     "SCALED_PSNR": torch.tensor(scaled_psnrs).mean().item(),
                                     "SCALED_LPIPS": torch.tensor(scaled_lpipss).mean().item(),
                                     })
        print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        with open(image_path + f"/{env_light}/results.json", 'w') as fp:
            json.dump(full_dict[env_light], fp, indent=True)
        with open(image_path + f"/{env_light}/per_view.json", 'w') as fp:
            json.dump(per_view_dict[env_light], fp, indent=True)
        # except:
        #     print("Unable to compute metrics for model", env_light)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--gt_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--img_paths', '-i', required=True, nargs="+", type=str, default=[])
    
    args = parser.parse_args()
    evaluate(args.gt_paths, args.img_paths)
