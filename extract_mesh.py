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
from fields.shape_renders import SDF_RENDER_DICT
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, TensoSDFOptimParams
from scene import GaussianModel

import numpy as np
import mcubes
import trimesh



def extract_fields(bound_min, bound_max, resolution, query_func, batch_size=64, outside_val=1.0):
    N = batch_size
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()
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

def reconstruct_sdf(model_path, scene, sdf_render, resolution=256):
    mesh_path = os.path.join(model_path, "meshes")
    name = 'sdf_mesh.ply'
    bbox_min, bbox_max = scene.gaussians.get_xyz.min(0)[0], scene.gaussians.get_xyz.max(0)[0]
    bbox_min = bbox_min * 1.1
    bbox_max = bbox_max * 1.1
    print('bbox:', bbox_min, bbox_max)
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

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, sdf_opt: TensoSDFOptimParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, pipeline.env_mode, dataset.envmap_res, 
                              dataset.use_delta, use_metallic=True)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        sdf_render = SDF_RENDER_DICT[sdf_opt.sdf_mode]({}).cuda()
        sdf_render.training_setup(sdf_opt)
        sdf_render.load_iter(scene.model_path, iteration, True)
        sdf_render.eval()
        reconstruct_sdf(dataset.model_path, scene, sdf_render)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    sdf_opt = TensoSDFOptimParams(parser)
    parser.add_argument("--iteration", default=6_000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=2048, type=int, help='Mesh: resolution for unbounded mesh extraction')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), sdf_opt.extract(args))