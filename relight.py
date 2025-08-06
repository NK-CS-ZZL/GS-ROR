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
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import RENDER_DICT, render_lighting
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
from utils.image_utils import apply_depth_colormap
from scene.NVDIFFREC.util import save_image_raw, latlong_to_cubemap2
import imageio
imageio.plugins.freeimage.download()
from utils.camera_utils import interpolate_camera


def render_lightings(model_path, name, iteration, scene):
    gaussians = scene.gaussians
    lighting_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(lighting_path, exist_ok=True)    
    lighting = render_lighting(gaussians)
    torchvision.utils.save_image(lighting, os.path.join(lighting_path, "00000.png"))
    save_image_raw(os.path.join(lighting_path, "00000.hdr"), lighting.permute(1,2,0).detach().cpu().numpy())

def render_set(model_path, name, iteration, views, scene, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_no_rescale_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_no_rescale")
    albedo_path = os.path.join(model_path, name, "ours_{}".format(iteration), "albedo")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gt_masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_mask")
    makedirs(render_path, exist_ok=True)
    makedirs(render_no_rescale_path, exist_ok=True)
    makedirs(albedo_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gt_masks_path, exist_ok=True)
    render_fn = RENDER_DICT[pipeline.gaussian_type]

    # rescale
    valid_albedo_factor = []
    print(views)
    if views[0].albedo_image is None:
        three_channel_ratio = torch.tensor([1,1,1]).cuda()
        print("No gt albedo available")
    else:
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            torch.cuda.synchronize()

            render_pkg = render_fn(view, scene, pipeline, background, debug=False, 
                                is_training=True)

            torch.cuda.synchronize()
            gt_alpha_mask = view.gt_alpha_mask
            gt_alpha_mask[gt_alpha_mask>=0.5] = 1
            gt_alpha_mask[gt_alpha_mask<0.5] = 0
            valid_albedo = render_pkg['albedo'].permute(1, 2, 0)[gt_alpha_mask.bool()[0]]
            valid_gt_albedo = view.albedo_image.permute(1, 2, 0)[gt_alpha_mask.bool()[0]]
            rescale_factor = valid_gt_albedo / (valid_albedo.cpu() + 1e-8)
            valid_albedo_factor.append(rescale_factor)
        valid_albedo_factor = torch.cat(valid_albedo_factor, dim=0)
        three_channel_ratio, _ = (valid_albedo_factor).median(dim=0)
    print('rescale factor is: ', three_channel_ratio)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()

        render_pkg = render_fn(view, scene, pipeline, background, debug=True, 
                            is_training=False,
                            rescale=three_channel_ratio.view(3, 1, 1).cuda())
        render_pkg_no_rescale = render_fn(view, scene, pipeline, background, debug=True, 
                            is_training=False
                            )

        torch.cuda.synchronize()

        gt = view.original_image[0:3, :, :]
        gt_alpha_mask = view.gt_alpha_mask
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(render_pkg_no_rescale["render"], os.path.join(render_no_rescale_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(render_pkg["albedo"], os.path.join(albedo_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt_alpha_mask, os.path.join(gt_masks_path, '{0:05d}'.format(idx) + ".png"))
        for k in render_pkg.keys():
            if render_pkg[k].dim()<3 or render_pkg[k].shape[0] < 1 or k=="render" or k=="delta_normal_norm":
                continue
            save_path = os.path.join(model_path, name, "ours_{}".format(iteration), k)
            makedirs(save_path, exist_ok=True)
            if k == "depth":
                render_pkg[k] = apply_depth_colormap(-render_pkg["depth"][0][...,None]).permute(2,0,1)
            elif "normal" in k:
                render_pkg[k] = 0.5 + (0.5*render_pkg[k])
                render_pkg[k] = render_pkg[k] * render_pkg["alpha"] + (1-render_pkg["alpha"])
            torchvision.utils.save_image(render_pkg[k], os.path.join(save_path, '{0:05d}'.format(idx) + ".png"))

def relight_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, 
                 skip_train : bool, skip_test : bool, relight_envmap : str, transform: str):
    envmap = imageio.imread(relight_envmap)
    name = os.path.splitext(os.path.basename(relight_envmap))[0]
    envmap = torch.tensor(envmap).cuda()[:, :, :3].contiguous()
    envmap[envmap.isinf()] = 0
    envmap[envmap<0] = 0
    transform_mat = None
    use_metallic = True
    if transform == 'tir':
        transform_mat = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.float32, device="cuda") # check!
        use_metallic = True
    elif transform == 'real':
        transform_mat = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.float32, device="cuda") 
    print('use_metallic', use_metallic)
    envmap = latlong_to_cubemap2(envmap, (512, 512), transform_mat)
    envmap = envmap.reshape(6, 512, 512, 3)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.env_mode, dataset.envmap_res,
                              dataset.use_delta, use_metallic=use_metallic)

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        scene.gaussians.set_envmap(envmap.contiguous())

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if args.interpolate > 0:
            cams = interpolate_camera(scene.getTrainCameras(), args.interpolate)
        else:
            cams = scene.getTrainCameras()
        if not skip_train:
             render_set(dataset.model_path + f"/relight/{name}", "train", scene.loaded_iter, cams, scene, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path + f"/relight/{name}", "test", scene.loaded_iter, scene.getTestCameras(), scene, pipeline, background)

        render_lightings(dataset.model_path + f"/relight/{name}", "lighting", scene.loaded_iter, scene)
             
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Relighting script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--envmap", type=str)
    parser.add_argument("--transform", type=str, default='nero')
    parser.add_argument("--interpolate", type=int, default=0)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    print('using transform mat: ', args.transform)
    relight_sets(model.extract(args), args.iteration, pipeline.extract(args), 
                 args.skip_train, args.skip_test, args.envmap, args.transform)