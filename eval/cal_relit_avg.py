import json
import numpy as np
import os
from argparse import ArgumentParser

parser = ArgumentParser(description="Relighting script parameters")
parser.add_argument("--root", default="", type=str)
args = parser.parse_args()
root = args.root
ssim = 0.0
psnr = 0.0
lpips = 0.0
scaled_psnr = 0.0
scaled_ssim = 0.0
scaled_lpips = 0.0
dirpaths = [dirpath for dirpath in os.listdir(root) if os.path.isdir(f'{root}/{dirpath}')]
total = len(dirpaths)
print(f'Root: {args.root}, Number of envmaps: {total}')
for dirpath in dirpaths:
    with open(f'{root}/{dirpath}/results.json', 'r') as fr:
        json_file = json.load(fr)
    ssim += json_file['SSIM']
    psnr += json_file['PSNR']
    lpips += json_file['LPIPS']
    if json_file.get('SCALED_PSNR') is not None:
        scaled_ssim += json_file['SCALED_SSIM']
        scaled_psnr += json_file['SCALED_PSNR']
        scaled_lpips += json_file['SCALED_LPIPS']

with open(f'{root}/relit_avg.json', 'w') as fw:
    json.dump({'PSNR': psnr / total,
               'SSIM': ssim / total,
               'LPIPS': lpips / total,
               'SCALED_PSNR': scaled_psnr / total,
               'SCALED_SSIM': scaled_ssim / total,
               'SCALED_LPIPS': scaled_lpips / total,
               }, fw)