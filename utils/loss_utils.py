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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils.image_utils import erode
import numpy as np

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1_, img2_, window_size=11, size_average=True):
    img1 = img1_.squeeze(0)
    img2 = img2_.squeeze(0)
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    img1 = img1[None,...]
    img2 = img2[None,...]
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def zero_one_loss(img):
    zero_epsilon = 1e-3
    val = torch.clamp(img, zero_epsilon, 1 - zero_epsilon)
    loss = torch.mean(torch.log(val) + torch.log(1 - val))
    return loss

def predicted_normal_loss(normal, normal_ref, weight=None):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    # normal: (3, H, W), normal_ref: (3, H, W), alpha: (3, H, W)
    # if alpha is not None:
    #     device = alpha.device
    #     weight = alpha.detach().cpu().numpy()[0]
    #     weight = (weight*255).astype(np.uint8)

    #     weight = erode(weight, erode_size=4)

    #     weight = torch.from_numpy(weight.astype(np.float32)/255.)
    #     weight = weight[None,...].repeat(3,1,1)
    #     weight = weight.to(device) 
    # else:
    #     weight = torch.ones_like(normal_ref)

    w = weight.permute(1,2,0).reshape(-1).detach()
    n = normal_ref.permute(1,2,0).reshape(-1,3).detach()
    n_pred = normal.permute(1,2,0).reshape(-1,3)
    loss = (w * (1.0 - torch.sum(n * n_pred, axis=-1))).mean()

    return loss

def delta_normal_loss(delta_normal_norm, weight=None):
    # delta_normal_norm: (3, H, W), alpha: (3, H, W)
    # if alpha is not None:
    #     device = alpha.device
    #     weight = alpha.detach().cpu().numpy()[0]
    #     weight = (weight*255).astype(np.uint8)

    #     weight = erode(weight, erode_size=4)

    #     weight = torch.from_numpy(weight.astype(np.float32)/255.)
    #     weight = weight[None,...].repeat(3,1,1)
    #     weight = weight.to(device) 
    # else:
    #     weight = torch.ones_like(delta_normal_norm)

    w = weight.permute(1,2,0).reshape(-1).detach()
    ch = delta_normal_norm.shape[0]
    l = delta_normal_norm.permute(1,2,0).reshape(-1,ch)[...,0]
    loss = (w * l).mean()

    return loss

def cam_depth2world_point(cam_z, pixel_idx, intrinsic, extrinsic):
    '''
    cam_z: (1, N)
    pixel_idx: (1, N, 2)
    intrinsic: (3, 3)
    extrinsic: (4, 4)
    world_xyz: (1, N, 3)
    '''
    valid_x = (pixel_idx[..., 0] + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0]
    valid_y = (pixel_idx[..., 1] + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1]
    ndc_xy = torch.stack([valid_x, valid_y], dim=-1)
    # inv_scale = torch.tensor([[W - 1, H - 1]], device=cam_z.device)
    # cam_xy = ndc_xy * inv_scale * cam_z[...,None]
    cam_xy = ndc_xy * cam_z[...,None]
    cam_xyz = torch.cat([cam_xy, cam_z[...,None]], dim=-1)
    world_xyz = torch.cat([cam_xyz, torch.ones_like(cam_xyz[...,0:1])], axis=-1) @ torch.inverse(extrinsic).transpose(0,1)
    world_xyz = world_xyz[...,:3]
    return world_xyz, cam_xyz

def tv_loss(x):
    def _tensor_size(t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])
    count_w = _tensor_size(x[:,:,:,1:])

    # count_w = max(self._tensor_size(x[:,:,:,1:]), 1)

    h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
    w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
    return 2*(h_tv/count_h+w_tv/count_w)/batch_size

def cal_gradient(data):
    """
    data: [1, C, H, W]
    """
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(data.device)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(data.device)

    weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    grad_x = F.conv2d(data, weight_x, padding='same')
    grad_y = F.conv2d(data, weight_y, padding='same')
    gradient = torch.abs(grad_x) + torch.abs(grad_y)

    return gradient

def bilateral_smooth_loss(data, image, mask):
    """
    image: [C, H, W]
    data: [C, H, W]
    mask: [C, H, W]
    """
    rgb_grad = cal_gradient(image.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)  # [1, H, W]
    data_grad = cal_gradient(data.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)  # [1, H, W]

    smooth_loss = (data_grad * (-rgb_grad).exp() * mask).mean()

    return smooth_loss

def base_smooth_loss(data, image, mask):
    """
    image: [C, H, W]
    data: [C, H, W]
    mask: [C, H, W]
    """
    data_grad = cal_gradient(data.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)  # [1, H, W]

    smooth_loss = (data_grad * mask).mean()

    return smooth_loss


def depth_smooth_loss(data, points, mask):
    """
    image: [C, H, W]
    data: [C, H, W]
    mask: [C, H, W]
    """
    dist_x = ((points[1:, :] - points[:-1, :])**2).sum(-1)
    dist_x[mask[0, 1:]<0.01] = 100000
    dist_y = ((points[:, 1:] - points[:, :-1])**2).sum(-1)
    dist_y[mask[0, :, 1:]<0.01] = 100000
    grad_x = data[:, 1:, :] - data[:, :-1, :]
    grad_y = data[:, :, 1:] - data[:, :, :-1]
    reweight_grad_x = grad_x * 1/(1+dist_x)
    reweight_grad_y = grad_y * 1/(1+dist_y)
    
    smooth_loss = reweight_grad_x.mean() + reweight_grad_y.mean()

    return smooth_loss

TOPK = 8
def knn_eik_v2(xyz, sdf, sdf_grad, topk_idx):
    sdf_grad = F.normalize(sdf_grad, dim=-1)
    topk_sdf = sdf[topk_idx.view(-1)].view(-1, TOPK, 1)
    topk_grad =  sdf_grad[topk_idx.view(-1)].view(-1, TOPK, 3)
    topk_xyz = xyz[topk_idx.view(-1)].view(-1, TOPK, 3)
    diff_sdf = topk_sdf - sdf.view(-1, 1, 1)
    dist_vec = topk_xyz - xyz.view(-1, 1, 3)
    dist = torch.norm(dist_vec, p=2, dim=-1)
    mask = torch.logical_and((dist.detach() < 0.1), (diff_sdf[...,0].detach() < 0.1))
    proj_d = torch.einsum("ijk,ik->ij", dist_vec, sdf_grad)
    cos_topk = torch.abs(torch.einsum("ijk,ik->ij", topk_grad, sdf_grad)).detach()
    return torch.mean(torch.abs(torch.abs(diff_sdf[...,0]) - torch.abs(proj_d))*mask * torch.abs(cos_topk)) + (torch.abs(diff_sdf.squeeze()) - dist).clamp(0).mean() * 10


def knn_eik_v4(xyz, sdf, sdf_grad, topk_idx):
    sdf_grad = F.normalize(sdf_grad, dim=-1)
    topk_sdf = sdf[topk_idx.view(-1)].view(-1, TOPK, 1)
    topk_grad =  sdf_grad[topk_idx.view(-1)].view(-1, TOPK, 3)
    topk_xyz = xyz[topk_idx.view(-1)].view(-1, TOPK, 3)
    diff_sdf = topk_sdf - sdf.view(-1, 1, 1)
    dist_vec = topk_xyz - xyz.view(-1, 1, 3)
    dist = torch.norm(dist_vec, p=2, dim=-1)
    mask = torch.logical_and((dist.detach() < 0.1), (diff_sdf[...,0].detach() < 0.1))
    proj_d = torch.einsum("ijk,ik->ij", dist_vec, sdf_grad)
    cos_topk = torch.abs(torch.einsum("ijk,ik->ij", topk_grad, sdf_grad)).detach()
    return torch.mean(torch.abs(diff_sdf[...,0] - proj_d.detach()) * mask * torch.abs(cos_topk)) + (torch.abs(diff_sdf.squeeze()) - dist.detach()).clamp(0).mean() * 10