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
from gaussian_renderer import RENDER_DICT
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
from utils.image_utils import apply_depth_colormap
from scene.NVDIFFREC.util import save_image_raw, latlong_to_cubemap2, linear2srgb_torch
import imageio
import numpy as np
imageio.plugins.freeimage.download()
from scene.cameras import Camera
import torch.nn.functional as F
from utils.graphics_utils import focal2fov, lookat2c2w
from tqdm import tqdm


def get_w2c(point):
  c2w = torch.eye(4).float()
  # up = torch.tensor([0, 1, 0]).float()
  up = torch.tensor([0, 0, 1]).float()
  ori = torch.tensor([0, 0, 0]).float()
  new_look_at = F.normalize(ori - point, 2, -1)
  up = F.normalize(up, p=2, dim=0)
  
  c2w_new = lookat2c2w(new_look_at, up)
  c2w[:3, :3] = c2w_new
  c2w[:3, 3] = point
  w2c = np.linalg.inv(c2w)
  R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
  T = w2c[:3, 3]
  return R, T

def gen_traj(num_views, y_range=[1, 2]):
  unit = 2 * np.pi / num_views
  y_unit = (y_range[1] - y_range[0]) / num_views * 2
  y_start = y_range[0]
  points = []
  flag = False
  for i in range(num_views):
    theta = unit * i
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    if i <= num_views // 2:
      y = y_start + y_unit * i
    else:
      if flag == False:
        y_start = y
        flag = True
      y = y_start - y_unit * (i - num_views // 2)
    r = np.sqrt(9 - y ** 2)
    x = sin_theta * r
    z = cos_theta * r
    # points.append(torch.tensor([x, y, z]).float())
    points.append(torch.tensor([x, z, y]).float() * 1.5) 
  return points
    
    
def render_set(model_path, iteration, views, scene, pipeline, background):
    render_path = os.path.join(model_path, "ours_{}".format(iteration), "renders")
    render_no_rescale_path = os.path.join(model_path, "ours_{}".format(iteration), "renders_no_rescale")
    albedo_path = os.path.join(model_path, "ours_{}".format(iteration), "albedo")
    render_fn = RENDER_DICT[pipeline.gaussian_type]
    makedirs(render_path, exist_ok=True)
    makedirs(render_no_rescale_path, exist_ok=True)
    makedirs(albedo_path, exist_ok=True)

    num_views = 360
    points = gen_traj(num_views, [1.5, 1.51])

    focal_x = 1111.111
    focal_y = 1111.111
    fovX = focal2fov(focal_x, 800)
    fovY = focal2fov(focal_y, 800)
    for idx in tqdm(range(num_views)):

        R, T = get_w2c(points[idx])
        view = Camera(colmap_id=idx, R=R, T=T, 
                    FoVx=fovX, FoVy=fovY, 
                    image=torch.ones((3, 800, 800)), 
                    gt_alpha_mask=torch.ones((800, 800)),
                    image_name=str(idx), uid=idx, 
                    data_device='cuda', 
                    normal_image=None,
                    albedo_image=None)
        render_pkg = render_fn(view, scene, pipeline, background, debug=False, 
                            is_training=True, render_radiance=False, 
                            radiance_only=False, defer_radiance=pipeline.defer_radiance,
                            )
        torch.cuda.synchronize()

        gt_alpha_mask = view.gt_alpha_mask
        mask = render_pkg['alpha'] > 0.9
        torchvision.utils.save_image(render_pkg["render"]*mask + (1.-mask.float()), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(render_pkg["albedo"], os.path.join(albedo_path, '{0:05d}'.format(idx) + ".png"))

        for k in render_pkg.keys():
            if render_pkg[k].dim()<3 or k=="render" or k=="delta_normal_norm":
                continue
            save_path = os.path.join(model_path, "ours_{}".format(iteration), k)
            makedirs(save_path, exist_ok=True)
            if k == "depth":
                render_pkg[k] = apply_depth_colormap(-render_pkg["depth"][0][...,None]).permute(2,0,1)
            elif "normal" in k:
                render_pkg[k] = 0.5 + (0.5*render_pkg[k])
                render_pkg[k] = render_pkg[k] * render_pkg['alpha'] + (1-render_pkg['alpha'])
            torchvision.utils.save_image(render_pkg[k], os.path.join(save_path, '{0:05d}'.format(idx) + ".png"))

    
def relight_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, 
                 relight_envmap : str, transform: str):
    envmap = imageio.imread(relight_envmap)
    name = os.path.splitext(os.path.basename(relight_envmap))[0]
    envmap = torch.tensor(envmap).cuda()[:, :, :3].contiguous()
    # if pipeline.render_mode.find('disney') == -1:
    #     envmap = linear2srgb_torch(envmap.clamp(0,1))
    envmap[envmap.isinf()] = 0
    envmap[envmap<0] = 0
    transform_mat = None
    if transform == 'tir':
        transform_mat = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.float32, device="cuda") # check!
    
    envmap = torch.ones_like(envmap) * 0.1
    envmap[:100,] = 2.0
    envmap = latlong_to_cubemap2(envmap, (512, 512), transform_mat)
    envmap = envmap.reshape(6, 512, 512, 3)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, pipeline.env_mode, dataset.envmap_res, 
                                  dataset.use_delta, use_metallic=True)

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        scene.gaussians.set_envmap(envmap.contiguous())

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path + f"/relight_free_w/{name}", scene.loaded_iter, scene.getTrainCameras(), scene, pipeline, background)
             
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Relighting script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--envmap", type=str)
    parser.add_argument("--transform", type=str, default='nero')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    relight_sets(model.extract(args), args.iteration, pipeline.extract(args), 
                 args.envmap, args.transform)