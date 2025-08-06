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
import torch.nn.functional as F
from random import randint
from torch.utils.tensorboard import SummaryWriter
from gaussian_renderer import RENDER_DICT
from fields.shape_renders import SDF_RENDER_DICT
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, TensoSDFOptimParams, OptimizationParams
from scene import GaussianModel
import numpy as np
import mcubes
import trimesh
from tqdm import tqdm
import imageio

def extract_fields(bound_min, bound_max, resolution, query_func, batch_size=64, outside_val=1.0):
    N = batch_size
    X = torch.linspace(bound_min[0], bound_max[0], resolution, device='cuda').split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution, device='cuda').split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution, device='cuda').split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).detach()
                    outside_mask = torch.norm(pts,dim=-1)>=1.0
                    val[outside_mask]=outside_val
                    val = val.reshape(len(xs), len(ys), len(zs)).cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, outside_val=1.0):
    u = extract_fields(bound_min, bound_max, resolution, query_func, outside_val=outside_val)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles

def reconstruct_sdf(model_path, mesh_dir, iteration, scene, sdf_render, resolution=256):
    mesh_path = os.path.join(model_path, mesh_dir)
    name = f'sdf_mesh_{iteration}.ply'
    bbox_min, bbox_max = scene.gaussians.get_xyz.min(0)[0], scene.gaussians.get_xyz.max(0)[0]
    bbox_min = bbox_min * 2
    bbox_max = bbox_max * 2
    def func(pts):
        level = sdf_render.compute_sample_level(pts).cuda()
        sdf = sdf_render.sdf_network.sdf(pts.reshape(-1, 3), level)
        sdf = sdf.cuda()
        return sdf
    with torch.no_grad():
        vertices, triangles = extract_geometry(bbox_min.cuda(), bbox_max.cuda(), resolution, 0, func)
    mesh = trimesh.Trimesh(vertices, triangles)
    os.makedirs(mesh_path, exist_ok=True)
    mesh.export(f"{mesh_path}/{name}")

def finetune_sdf(args, dataset : ModelParams, iteration : int, pipe : PipelineParams, 
                 opt: OptimizationParams, sdf_opt: TensoSDFOptimParams):
    gaussians = GaussianModel(dataset.sh_degree, pipe.env_mode, dataset.envmap_res, 
                            dataset.use_delta, use_metallic=True)
    tb_writer = SummaryWriter(f"{args.model_path}/sdf_log")
    render_fn = RENDER_DICT[pipe.gaussian_type]
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    sdf_render = SDF_RENDER_DICT[sdf_opt.sdf_mode](sdf_opt.__dict__).cuda()
    sdf_render.load_iter(scene.model_path, iteration)
    sdf_render.deviation_network.variance = torch.nn.Parameter(torch.tensor(0.3, device='cuda', dtype=torch.float32))
    sdf_render.training_setup(sdf_opt)
    sdf_render.train()
    # sdf_render.updateAlphaMask()
    viewpoint_stack = None
    batch_size = 2048
    pbar = tqdm(range(0, 6001))
    mean_loss = 0
    for iteration in pbar: 
        if dataset.random_background == True:
            background = torch.rand((3)).cuda()
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) 
        mask = viewpoint_cam.gt_alpha_mask.cuda()
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        if gaussians.env_mode=="envmap":
            gaussians.brdf_mlp.build_mips()
        sdf_losses = {}
        render_pkg = render_fn(viewpoint_cam, scene, pipe, background, 
                            debug=False, is_training=True, render_radiance=pipe.render_radiance, 
                            radiance_only=False, defer_radiance=pipe.defer_radiance, attr_only=True)
        normal = render_pkg['normal'].permute(1, 2, 0).detach()
        acc = render_pkg['alpha'].permute(1, 2, 0).detach()

        viewdirs, valid_mask = viewpoint_cam.get_filtered_ray()
        valid_normal = normal.view(-1, 3)
        valid_viewdirs = viewdirs.view(-1, 3)
        valid_acc = acc.view(-1, 1)
        bs = valid_viewdirs.shape[0]
        if  bs > batch_size:
            idx = torch.randint(bs, [batch_size])
            valid_viewdirs = valid_viewdirs[idx]
            valid_normal = valid_normal[idx]
            valid_mask = mask.permute(1, 2, 0).view(-1, 1)[idx]
            valid_acc = valid_acc[idx]
            default_normal = torch.tensor([[0, 0, 1]]).float().cuda()
            valid_normal = valid_normal * valid_acc + (1-valid_acc) * default_normal
            valid_normal = F.normalize(valid_normal, dim=-1) 
            bs = batch_size
        ray_batch = {
            'rays_o': viewpoint_cam.camera_center.repeat(bs, 1),
            'dirs': valid_viewdirs, 
            'step': iteration+1000,
            'masks': valid_mask,
            'bg': background[None, :],
            'normal': valid_normal,
        }
        tensosdf_output = sdf_render(ray_batch,)
        loss = torch.tensor(0.).cuda().float()
        for key in tensosdf_output.keys():
            if key.find('loss') > -1:
                sdf_loss = tensosdf_output[key].mean()* getattr(sdf_opt, f'lambda_'+key[5:])
                sdf_losses[key] = sdf_loss 
                loss += sdf_loss
        sdf_render.optimizer.zero_grad()
        loss.backward()
        sdf_render.optimizer.step()
        mean_loss = 0.8 * mean_loss + 0.2 * loss.item()
        pbar.set_description_str(f'Iteration {iteration:05d} Loss {mean_loss:.4f}')
        for sdf_loss in sdf_losses.keys():
            tb_writer.add_scalar(sdf_loss, sdf_losses[sdf_loss].item(), iteration)
        tb_writer.add_scalar('total_loss', loss.item(), iteration)
        if iteration == 2000:
            sdf_render.upsample_sdf_grid(torch.tensor([512, 512, 512]))
            sdf_render.training_setup(sdf_opt)
        sdf_render.update_learning_rate(iteration)
        tb_writer.add_scalar('learning_rate', sdf_render.optimizer.param_groups[0]['lr'], iteration)
        if iteration % 3000 == 0:
            sdf_render.save(os.path.join(scene.model_path, 'finetuned'), iteration)
            reconstruct_sdf(dataset.model_path, args.mesh_dir, iteration, scene, sdf_render, resolution=512)
            with torch.no_grad():
                viewpoint = viewpoint_cam
                viewdirs, valid_mask = viewpoint_cam.get_filtered_ray()
                valid_viewdirs = viewdirs.view(-1, 3)
                gt_image = viewpoint.original_image.cuda()
                sdf_normal = torch.zeros((*gt_image.shape[1:], 3), device='cuda', dtype=torch.float32)
                bs = valid_viewdirs.shape[0]
                chunk_idxs = torch.split(torch.arange(bs), batch_size) 
                outputs_dict = {}
                for chunk_idx in chunk_idxs:
                    ray_batch = {
                                    'rays_o': viewpoint.camera_center.repeat(len(chunk_idx), 1),
                                    'dirs': valid_viewdirs[chunk_idx], 
                                    'step': iteration + 1000,
                                    'masks': valid_mask[chunk_idx]
                                }
                    tensosdf_output = sdf_render(ray_batch, is_train=False)
                    for k in tensosdf_output.keys():
                        if k in outputs_dict:
                            outputs_dict[k].append(tensosdf_output[k].detach().clone())
                        else:
                            outputs_dict[k] = [tensosdf_output[k].detach().clone()]
                    torch.cuda.empty_cache()
                    sdf_render.optimizer.zero_grad()
                for k in outputs_dict.keys():
                    if k in ['normal']:
                        outputs_dict[k] = torch.cat(outputs_dict[k], dim=0)
                sdf_normal = outputs_dict['normal'].detach().reshape(800, 800, 3)
                sdf_normal = (sdf_normal * 255. / 2 + 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
                os.makedirs(f'{dataset.model_path}/sdf_vis', exist_ok=True)
                imageio.imwrite(f'{dataset.model_path}/sdf_vis/normal_{iteration}.png', sdf_normal)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipe = PipelineParams(parser)
    opt = OptimizationParams(parser)
    sdf_opt = TensoSDFOptimParams(parser)
    parser.add_argument("--load_iteration", default=24_000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=2048, type=int, help='Mesh: resolution for unbounded mesh extraction')
    parser.add_argument("--mesh_dir", default='mesh_test', type=str, help='Mesh: directory to save mesh')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    finetune_sdf(args, model.extract(args), args.load_iteration, pipe.extract(args), opt.extract(args), sdf_opt.extract(args))