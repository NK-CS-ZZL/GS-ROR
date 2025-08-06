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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, get_rays

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", 
                 normal_image=None, albedo_image=None, scene_name=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        
        self.image_name = image_name
        self.ray_cache = None
        self.ray_mask = None
        self.scene_name = scene_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.original_normal_image = None
        if normal_image is not None:
            self.original_normal_image = normal_image.clamp(0.0, 1.0).to(self.data_device)
            original_normal_image = self.original_normal_image * 2 - 1. # (0, 1) -> (-1, 1)
            original_normal_image = torch.nn.functional.normalize(original_normal_image, p=2, dim=0)
            fg_mask = torch.where((self.original_normal_image==0.).sum(0)<3, True, False)[None,...].repeat(3, 1, 1)
            original_normal_image = torch.where(fg_mask, original_normal_image, torch.ones_like(original_normal_image))            
            self.original_normal_image = original_normal_image
        self.albedo_image = None
        if albedo_image is not None:
            self.albedo_image = albedo_image
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.gt_alpha_mask = gt_alpha_mask
        self.focal_x = fov2focal(FoVx, self.image_width)
        self.focal_y = fov2focal(FoVy, self.image_height)

        # if gt_alpha_mask is not None:
        if False:
            self.original_image *= gt_alpha_mask.to(self.data_device)
            if self.original_normal_image is not None:
                self.original_normal_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            if self.original_normal_image is not None:
                    self.original_normal_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def get_calib_matrix_nerf(self):
        focal = fov2focal(self.FoVx, self.image_width)  # original focal length
        intrinsic_matrix = torch.tensor([[focal, 0, self.image_width / 2], [0, focal, self.image_height / 2], [0, 0, 1]]).float()
        c2w = self.world_view_transform.inverse().T # cam2world
        return intrinsic_matrix, c2w

    def get_rays(self):
        if self.ray_cache is not None:
            return self.ray_cache
        c2w = self.world_view_transform.inverse().T
        focal_X = fov2focal(self.FoVx, self.image_width)
        focal_Y = fov2focal(self.FoVy, self.image_height)
        viewdirs = get_rays(self.image_width, self.image_height, [focal_X, focal_Y], c2w[:3,:3])
        self.ray_cache = viewdirs.view(-1, 3)
        return viewdirs.view(-1, 3)
    
    def get_filtered_ray(self):
        if self.ray_cache is None:
            self.filtering_train_rays()
        return self.ray_cache, self.ray_mask
    
    @torch.no_grad()
    def filtering_train_rays(self):
        rays_d = self.get_rays().view(-1, 3)
        N = rays_d.shape[0]
        rays_o = self.camera_center.repeat(N, 1)
        # aabb = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]).cuda()
        aabb = torch.tensor([[-50., -50., -50.], [50., 50., 50.]]).cuda()

        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (aabb[1] - rays_o) / vec
        rate_b = (aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1)
        t_max = torch.maximum(rate_a, rate_b).amin(-1)
        mask_inbbox = t_max > t_min
        self.ray_mask = mask_inbbox        
        self.ray_cache = rays_d

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

