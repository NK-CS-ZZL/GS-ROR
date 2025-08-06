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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import RENDER_DICT, render_lighting
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import GaussianModel
from utils.image_utils import apply_depth_colormap
from scene.NVDIFFREC.util import save_image_raw
from utils.camera_utils import interpolate_camera

# debug

def render_lightings(model_path, name, iteration, scene):
    gaussians = scene.gaussians
    lighting_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(lighting_path, exist_ok=True)    
    lighting = render_lighting(gaussians)
    torchvision.utils.save_image(lighting, os.path.join(lighting_path, "00000.png"))
    save_image_raw(os.path.join(lighting_path, "00000.hdr"), lighting.permute(1,2,0).detach().cpu().numpy())

def render_set(model_path, name, iteration, views, scene, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gt_masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_mask")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gt_masks_path, exist_ok=True)
    render_fn = RENDER_DICT[pipeline.gaussian_type]

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()
        render_pkg = render_fn(view, scene, pipeline, background, debug=False, 
                                is_training=True, render_radiance=False, 
                            radiance_only=False, defer_radiance=pipeline.defer_radiance)

        torch.cuda.synchronize()

        gt = view.original_image[0:3, :, :]
        gt_alpha_mask = view.gt_alpha_mask
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt_alpha_mask, os.path.join(gt_masks_path, '{0:05d}'.format(idx) + ".png"))
        for k in render_pkg.keys():
            if render_pkg[k].dim()<3 or k=="render" or k=="delta_normal_norm":
                continue
            save_path = os.path.join(model_path, name, "ours_{}".format(iteration), k)
            makedirs(save_path, exist_ok=True)
            if k == "alpha":
                render_pkg[k] = apply_depth_colormap(render_pkg["alpha"][0][...,None], min=0., max=1.).permute(2,0,1)
            if k == "depth":
                render_pkg[k] = apply_depth_colormap(-render_pkg["depth"][0][...,None]).permute(2,0,1)
            elif "normal" in k:
                render_pkg[k] = 0.5 + (0.5*render_pkg[k])
            torchvision.utils.save_image(render_pkg[k], os.path.join(save_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, pipeline.env_mode, dataset.envmap_res, 
                                  dataset.use_delta, True)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        log_f = open(f'{dataset.model_path}/render_log.txt', 'w')
        log_f.write(f"Number of gaussians: {gaussians.get_xyz.shape[0]}\n")
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if args.interpolate > 0:
            cams = interpolate_camera(scene.getTrainCameras(), args.interpolate)
        else:
            cams = scene.getTrainCameras()
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, cams, scene, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene, pipeline, background)

        render_lightings(dataset.model_path, "lighting", scene.loaded_iter, scene)
        log_f.close() 

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--interpolate", type=int, default=0)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)