import os
import time
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import nvdiffrast.torch as dr
from fields.network_utils import (
    get_embedder, make_predictor_3layer, linear_to_srgb, contraction, sample_pdf,
    TVLoss, GaussianBlur1D, GaussianBlur2D, AlphaGridMask, InitSDFRegLoss
)
from torch.optim import Adam
from arguments import TensoSDFOptimParams

class TensoSDF(nn.Module):
    def __init__(self, gridSize, aabb, device='cuda', sdf_n_comp=36, 
                 sdf_dim = 256, app_dim = 128, init_n_levels = 3):
        super().__init__()
        self.sdf_n_comp = sdf_n_comp
        self.sdf_dim = sdf_dim
        self.app_dim = app_dim
        self.device = device

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]
        self.nplane = len(self.vecMode)
        self.init_radius = 0.2

        self.kernel_size = 5
        self.sigma = 0.5
        self.define_gaussian(self.kernel_size, self.sigma)

        self.update_gridSize_aabb(gridSize, aabb, init_n_levels)

        self.init_svd_volume(device)
        self.init_mlp(device, sdf_multires=3, sdf_feat_multires=0)


    def define_gaussian(self, kernel_size=5, sigma=0.8, stride=1):
        print(f"Gaussian settings: {kernel_size}, {sigma}")
        self.gaussian1d = GaussianBlur1D(kernel_size=kernel_size, sigma=sigma, stride=stride)
        self.gaussian2d = GaussianBlur2D(kernel_size=kernel_size, sigma=sigma, stride=stride)

    def gaussian_conv(self):
        self.sdf_plane_gaussian, self.sdf_line_gaussian = [], []
        for i in range(self.nplane):
            self.sdf_plane_gaussian.append(self.gaussian2d(self.sdf_plane[i].permute(1,0,2,3)).permute(1,0,2,3))
            self.sdf_line_gaussian.append(self.gaussian1d(self.sdf_line[i].permute(1,0,2,3).squeeze(-1)).unsqueeze(-1).permute(1,0,2,3))

    def update_gridSize_aabb(self, gridSize, aabb, n_levels):
        self.gridSize = gridSize
        self.aabb = aabb
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.units = self.aabbSize / (self.gridSize - 1)
        self.n_levels = n_levels
        print(f"current levels : {self.n_levels}, current units : {self.units}")

    def init_svd_volume(self, device):
        self.sdf_plane, self.sdf_line = self.circle_init_one_svd(self.sdf_n_comp, device)

    def init_mlp(self, device, sdf_multires, sdf_feat_multires=0):
        self.embed_fn = None
        sdf_input_ch = 3
        if sdf_multires > 0:
            self.embed_fn, sdf_input_ch = get_embedder(sdf_multires, input_dims=sdf_input_ch)

        self.sdf_feat_embed_fn = None
        sdf_feat_input_ch = self.sdf_n_comp * self.nplane
        if sdf_feat_multires > 0:
            self.sdf_feat_embed_fn, sdf_feat_input_ch = get_embedder(sdf_feat_multires, input_dims=sdf_feat_input_ch)
        
        self.sdf_mat = nn.Sequential(
            nn.Linear(sdf_feat_input_ch + sdf_input_ch, self.sdf_dim), nn.Softplus(beta=100),
            nn.Linear(self.sdf_dim , 1 + self.app_dim)
        ).to(device)

        torch.nn.init.constant_(self.sdf_mat[0].bias, 0.0)
        torch.nn.init.normal_(self.sdf_mat[0].weight, 0.0, np.sqrt(2) / np.sqrt(self.sdf_dim))
        torch.nn.init.constant_(self.sdf_mat[-1].bias, -self.init_radius)
        torch.nn.init.normal_(self.sdf_mat[-1].weight, mean=np.sqrt(np.pi) / np.sqrt(self.sdf_dim), std=0.0001)

    def init_mat_mlp(self, mat_multires=6):
        out_dim = 3 + 1 + 1
        self.mat_pos_embed_fn = None
        mat_pos_input_ch = 3
        if mat_multires > 0:
            self.mat_pos_embed_fn, mat_pos_input_ch = get_embedder(mat_multires, input_dims=mat_pos_input_ch)
        self.material_mlp = make_predictor_3layer(self.app_dim + mat_pos_input_ch, out_dim, run_dim=128)
         
    def circle_init_one_svd(self, n_component, device):
        plane_coef, line_coef = [], []
        for i in range(self.nplane):
            planeSize = self.gridSize[self.matMode[i]]
            lineSize = self.gridSize[self.vecMode[i]]
            init_plane = self.circle_init(planeSize).expand(n_component, planeSize[0], planeSize[1]).unsqueeze(0) # 1, n, grid, grid
            init_line = torch.ones((1, n_component, lineSize, 1)) * (1./(n_component * self.nplane)) # 1, n, grid, 1
            plane_coef.append(nn.Parameter(init_plane.clone()))
            line_coef.append(nn.Parameter(init_line.clone()))
        
        return nn.ParameterList(plane_coef).to(device), nn.ParameterList(line_coef).to(device)

    def circle_init(self, gridSize):
        x = torch.linspace(-1, 1, gridSize[0])
        y = torch.linspace(-1, 1, gridSize[1])
        x, y = torch.meshgrid(x, y)
        pts = torch.stack([x, y], dim=-1)
        init_sdf = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True) - self.init_radius
        return init_sdf.permute(2, 0, 1)  # (1, grid_sz, grid_sz)   
        
    def TV_loss_sdf(self, reg):
        total = 0
        for i in range(self.nplane):
            total += reg(self.sdf_plane[i])
            total += reg(self.sdf_line[i])
        return total

    def TV_loss_app(self, reg):
        return torch.zeros(1)
    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.sdf_line, 'lr': lr_init_spatialxyz}, 
                     {'params': self.sdf_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.sdf_mat.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def sdf(self, xyz_sampled, level_vol=None):
        return self.forward(xyz_sampled, level_vol)[..., :1]

    def sdf_hidden_appearance(self, xyz_sampled, level_vol):
        return self.forward(xyz_sampled, level_vol)[..., 1:]

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))
        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        new_levels = self.n_levels + 1
        res_target = ((res_target / 2**(new_levels - 1)).int() * 2**(new_levels - 1)) # can be divided by new_levels - 1
        self.sdf_plane, self.sdf_line = self.up_sampling_VM(self.sdf_plane, self.sdf_line, res_target)

        self.update_gridSize_aabb(res_target, self.aabb, new_levels)
        print(f'upsamping to {res_target}, remember to update renderer')
        return res_target, self.n_levels

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units

        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(self.nplane):
            mode0 = self.vecMode[i]
            self.sdf_line[i] = torch.nn.Parameter(
                self.sdf_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.sdf_plane[i] = torch.nn.Parameter(
                self.sdf_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )

        newSize = b_r - t_l
        self.update_gridSize_aabb(newSize, new_aabb)

        print(f'Shrink to {new_aabb}, remember to update renderer')
        return (newSize[0], newSize[1], newSize[2])
    
    @torch.no_grad()
    def compute_sample_level(self, x, k = 0.1):
        sdf = self.sdf(x)
        eps = self.units
        eps_x = torch.tensor([eps[0], 0., 0.], dtype=x.dtype, device=x.device)  # [3]
        eps_y = torch.tensor([0., eps[1], 0.], dtype=x.dtype, device=x.device)  # [3]
        eps_z = torch.tensor([0., 0., eps[2]], dtype=x.dtype, device=x.device)  # [3]
        sdf_x_pos = self.sdf(x + eps_x)  # [...,1]
        sdf_x_neg = self.sdf(x - eps_x)  # [...,1]
        sdf_y_pos = self.sdf(x + eps_y)  # [...,1]
        sdf_y_neg = self.sdf(x - eps_y)  # [...,1]
        sdf_z_pos = self.sdf(x + eps_z)  # [...,1]
        sdf_z_neg = self.sdf(x - eps_z)  # [...,1]
        delta_xx = (sdf_x_pos + sdf_x_neg - 2 * sdf)
        delta_yy = (sdf_y_pos + sdf_y_neg - 2 * sdf)
        delta_zz = (sdf_z_pos + sdf_z_neg - 2 * sdf)
        delta_mean = torch.sum(torch.cat([delta_xx, delta_yy, delta_zz], dim=-1).abs(), dim=-1, keepdim=True) / (3 * eps.mean()) # [...,1]
        return torch.clamp(1. - delta_mean, min=0., max=1.)
        
    def gradient(self, x, level_vol, training=False, sdf=None):
        eps = self.units
        # 1st-order gradient
        eps_x = torch.tensor([eps[0], 0., 0.], dtype=x.dtype, device=x.device)  # [3]
        eps_y = torch.tensor([0., eps[1], 0.], dtype=x.dtype, device=x.device)  # [3]
        eps_z = torch.tensor([0., 0., eps[2]], dtype=x.dtype, device=x.device)  # [3]
        sdf_x_pos = self.sdf(x + eps_x, level_vol)  # [...,1]
        sdf_x_neg = self.sdf(x - eps_x, level_vol)  # [...,1]
        sdf_y_pos = self.sdf(x + eps_y, level_vol)  # [...,1]
        sdf_y_neg = self.sdf(x - eps_y, level_vol)  # [...,1]
        sdf_z_pos = self.sdf(x + eps_z, level_vol)  # [...,1]
        sdf_z_neg = self.sdf(x - eps_z, level_vol)  # [...,1]
        gradient_x = (sdf_x_pos - sdf_x_neg) / (2 * eps[0])
        gradient_y = (sdf_y_pos - sdf_y_neg) / (2 * eps[1])
        gradient_z = (sdf_z_pos - sdf_z_neg) / (2 * eps[2])
        gradients = torch.cat([gradient_x, gradient_y, gradient_z], dim=-1)  # [...,3]
        # 2nd-order gradient (hessian)
        if training:
            assert sdf is not None  # computed when feed-forwarding through the network
            hessian_xx = (sdf_x_pos + sdf_x_neg - 2 * sdf) / (eps[0] ** 2)  # [...,1]
            hessian_yy = (sdf_y_pos + sdf_y_neg - 2 * sdf) / (eps[1] ** 2)  # [...,1]
            hessian_zz = (sdf_z_pos + sdf_z_neg - 2 * sdf) / (eps[2] ** 2)  # [...,1]
            hessian = torch.cat([hessian_xx, hessian_yy, hessian_zz], dim=-1)  # [...,3]
            normal_hessian = (gradients * hessian).sum(dim=-1) / (torch.sum(gradients ** 2, dim=-1) + 1e-5)
        else:
            normal_hessian = None
        return gradients, normal_hessian

    def forward(self, xyz_sampled, level_vol):
        # xyz_sampled : (rn*sn, 3)
        # plane + line basis
        xyz_sampled = contraction(xyz_sampled, self.aabb.to(xyz_sampled.device)).reshape(-1, 3)
        level = (torch.zeros([xyz_sampled.shape[0], 1], device=xyz_sampled.device) if level_vol is None else level_vol).view(-1, 1).unsqueeze(0).contiguous() # 1, N, 1
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)          # 3, rn * sn, 1, 2

        plane_coef_point,line_coef_point = [],[]
        planes, lines = self.sdf_plane, self.sdf_line
        for idx in range(self.nplane):
            plane_coef_point.append(
                dr.texture(planes[idx].permute(0, 2, 3, 1).contiguous(), 
                           coordinate_plane[[idx]], 
                           mip_level_bias=level, 
                           boundary_mode="clamp", 
                           max_mip_level=self.n_levels-1
                           ).permute(0, 3, 1, 2).contiguous().view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(
                dr.texture(lines[idx].permute(0, 2, 3, 1).contiguous(), 
                           coordinate_line[[idx]], 
                           mip_level_bias=level, 
                           boundary_mode="clamp", 
                           max_mip_level=self.n_levels-1
                           ).permute(0, 3, 1, 2).contiguous().view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        sigma_feature = plane_coef_point * line_coef_point

        inputs_xyz = xyz_sampled
        inputs_feat = sigma_feature.T
        if self.embed_fn is not None:
            inputs_xyz = self.embed_fn(xyz_sampled)
        if self.sdf_feat_embed_fn is not None:
            inputs_feat = self.sdf_feat_embed_fn(sigma_feature.T)
        out_feats = self.sdf_mat(torch.cat([inputs_feat, inputs_xyz], dim=-1))       # (rn * sn, 1)
        return out_feats

    def grid_gaussian_loss(self):
        total_loss = 0.
        k = self.kernel_size // 2
        for i in range(self.nplane):
            plane_gaussian = self.gaussian2d(self.sdf_plane[i].permute(1,0,2,3)).permute(1,0,2,3)
            line_gaussian = self.gaussian1d(self.sdf_line[i].permute(1,0,2,3).squeeze(-1)).unsqueeze(-1).permute(1,0,2,3)
            total_loss += torch.sum((self.sdf_plane[i][..., k:-k, k:-k] - plane_gaussian[..., k:-k, k:-k]).square())
            total_loss += torch.sum((self.sdf_line[i][..., k:-k, :] - line_gaussian[..., k:-k, :]).square())
        return total_loss

    def predict_normals(self, feats, xyz_sampled):
        inputs_xyz = xyz_sampled
        if self.normal_pos_embed_fn is not None:
            inputs_xyz = self.normal_pos_embed_fn(xyz_sampled)
        pred_normals = self.normal_mat(torch.cat([feats, inputs_xyz], dim=-1))
        return pred_normals
    
    def normal_denoiser(self, normal_noisy):
        pred_normals = self.normal_mat(self.normal_pos_embed_fn(normal_noisy))
        return pred_normals        
    
    def predict_materials(self, feats, xyz_sampled):
        inputs_xyz = xyz_sampled
        if self.mat_pos_embed_fn is not None:
            inputs_xyz = self.mat_pos_embed_fn(inputs_xyz)
        materials = self.material_mlp(torch.cat([feats, xyz_sampled], -1))
        albedo, roughness, metallic = materials[..., :3], materials[..., 3:4], materials[..., 4:]        
        return albedo, roughness, metallic
    
class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val, activation='exp'):
        super(SingleVarianceNetwork, self).__init__()
        self.act = activation
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        if self.act=='exp':
            return torch.ones([*x.shape[:-1], 1]).cuda() * torch.exp(self.variance * 10.0).cuda()
        elif self.act=='linear':
            return torch.ones([*x.shape[:-1], 1]) * self.variance * 10.0
        elif self.act=='square':
            return torch.ones([*x.shape[:-1], 1]) * (self.variance * 10.0) ** 2
        else:
            raise NotImplementedError

    def warp(self, x, inv_s):
        return torch.ones([*x.shape[:-1], 1]) * inv_s

class TensoSDFShapeRenderer(nn.Module):
    default_cfg = {
        # standard deviation for opacity density
        'std_net': 'default',
        'std_act': 'exp',
        'inv_s_init': 0.3,
        # 'freeze_inv_s_step': 1000,
        'freeze_inv_s_step': 3000,

        # geometry network
        'sdf_net': 'default',
        'sdf_activation': 'none',
        'sdf_bias': 0.5,
        'sdf_n_layers': 8,
        'sdf_freq': 6,
        'sdf_d_out': 129,
        'geometry_init': True,

        # shader network
        'shader_config': {},

        # sampling strategy
        'n_samples': 64,
        'n_bg_samples': 0,
        'inf_far': 1000.0,
        'n_importance': 64,
        'up_sample_steps': 4,  # 1 for simple coarse-to-fine sampling
        'perturb': 1.0,
        'anneal_end': 10000,
        # 'anneal_end': 5000,
        'train_ray_num': 1024,
        'test_ray_num': 2048,
        'clip_sample_variance': True,

        # dataset
        'database_name': 'nerf_synthetic/lego/black_800',

        # validation
        'test_downsample_ratio': True,
        'downsample_ratio': 0.25,
        'val_geometry': False,

        # losses
        'rgb_loss': 'charbonier',
        'apply_occ_loss': True,
        'apply_tv_loss' : True,
        'apply_sparse_loss' : True,
        'apply_hessian_loss': True,
        'apply_gaussian_loss': True,
        'occ_loss_step': 20000,
        'occ_loss_max_pn': 2048,
        'occ_sdf_thresh': 0.01,
        'gaussianLoss_step': 0,

        "fixed_camera": False,
        
        # Tenso
        'device' : 'cuda',
        'gridSize' : [128, 128, 128],
        'aabb' : [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]],
        'step_ratio' : 2.5,
        'alphaMask_thres' : 0.0001,
        'marched_weights_thres' : 0.0001,
        'sdf_n_comp' : 36,
        'app_n_comp' : 36,
        'sdf_dim' : 256,
        'app_dim' : 128,  
        'max_levels': 1,
        'has_radiance_field': False,
        'radiance_field_step': 0,
        'predict_BG': False,
        'isBGWhite': True,

        # dataset
        'nerfDataType': False,
        'split_manul': False,
        'apply_mask_loss': True,

        # alphaMask multi Length
        'mul_length': 10,
        'N_voxel_init': 2097152,
        'N_voxel_final': 64000000,
        'upsample_list': [5000, 10000],
        
        'lr_decay_iters': 30000,
        'lr_decay_target_ratio': 0.05,
        
    }

    def __init__(self, cfg):
        super().__init__()
        self.cfg = {**self.default_cfg, **cfg}
        print(cfg, self.default_cfg)
        self.device = self.cfg['device']
        if isinstance(self.cfg['gridSize'], list) is False:
            self.cfg['gridSize'] = [self.cfg['gridSize']]*3 
        gridSize = torch.tensor(self.cfg['gridSize'])
        max_levels = self.cfg['max_levels']
        self.aabb = torch.tensor(self.cfg['aabb'], device=self.device)
        self.center = torch.mean(self.aabb, axis=0).float().view(1, 1, 3)
        self.radius = (self.aabb[1] - self.center).mean().float()
        self.alphaMask = None
        self.step_ratio = self.cfg['step_ratio']
        self.alphaMask_thres = self.cfg['alphaMask_thres']
        self.marched_weights_thres = self.cfg['marched_weights_thres']
        self.sdf_n_comp, self.app_n_comp = self.cfg['sdf_n_comp'], self.cfg['app_n_comp']
        self.sdf_dim, self.app_dim = self.cfg['sdf_dim'], self.cfg['app_dim']
        self.lr_factor = self.pre_lr_factor = 1.0
        self.update_stepSize(gridSize, max_levels)
        print(gridSize)
        self.sdf_network = TensoSDF(
            self.gridSize, self.aabb, device=self.device, init_n_levels=self.max_levels,
            sdf_n_comp=self.sdf_n_comp, sdf_dim=self.sdf_dim, app_dim=self.app_dim)
        
        self.tv_reg = TVLoss()
        self.deviation_network = SingleVarianceNetwork(init_val=self.cfg['inv_s_init'], 
                                                       activation=self.cfg['std_act'])

        # background nerf is a nerf++ model (this is outside the unit bounding sphere, so we call it outer nerf)
        self.sdf_inter_fun = lambda x: self.sdf_network.sdf(x, None)
        self.N_voxel_list = (torch.round(torch.exp(
                torch.linspace(np.log(self.cfg['N_voxel_init']), 
                               np.log(self.cfg['N_voxel_final']), 
                               len(self.cfg['upsample_list']) + 1 if self.cfg['upsample_list'] is not None else 1))).long()).tolist()[1:]
        self.reset_times = 0
        
    def training_setup(self, training_args: TensoSDFOptimParams, decay=False):
        if decay:
            ratio = 2 ** self.reset_times
        else:
            ratio = 1
            
        l = self.get_train_opt_params(training_args.lr_xyz_init / ratio, 
                                      training_args.lr_net_init)
        print("lr_xyz", training_args.lr_xyz_init / ratio, "lr_net",  training_args.lr_net_init)
        self.optimizer = Adam(l, betas=(0.9, 0.99))
        self.sdf_init_loss = InitSDFRegLoss()
        self.reset_times += 1
        self.cfg['lr_decay_iters'] = training_args.lr_decay_iters
        self.cfg['lr_decay_target_ratio'] = training_args.lr_decay_target_ratio
    
    def N_to_reso(self, n_voxels, bbox):
        bbox = torch.tensor(bbox)
        xyz_min, xyz_max = bbox
        voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / 3)   # total volumes / number
        return ((xyz_max - xyz_min) / voxel_size).long().tolist()
    
    def update_stepSize(self, gridSize, max_levels):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)        
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize = torch.tensor(gridSize.cpu(), dtype=torch.int32).to(self.device)
        self.max_levels = max_levels
        self.units = self.aabbSize / (self.gridSize-1)
        self.stepSize = torch.mean(self.units)*self.step_ratio
        print("sampling step size: ", self.stepSize)
    
    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(128,128,128)):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        alpha, grid_xyz = self.compute_gridAlpha(gridSize)
        grid_xyz = grid_xyz.transpose(0,2).contiguous()
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None] # (1,1,gridSize012,)
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = grid_xyz[alpha>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))

        torch.set_default_tensor_type('torch.FloatTensor')
        return new_aabb

    @torch.no_grad()    
    def compute_gridAlpha(self, gridSize=None):
        gridSize = self.gridSize if gridSize is None else torch.LongTensor(gridSize).to(self.device)
        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        grid_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples
        stepLength = torch.mean(self.aabbSize / (gridSize - 1))
        alpha = torch.zeros_like(grid_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_grid_alpha(grid_xyz[i].view(-1,3), stepLength).view((gridSize[1], gridSize[2]))
        return alpha, grid_xyz

    def compute_grid_alpha(self, xyz_locs, length):
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
        
        alpha = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        if alpha_mask.any():
            xyz_sampled = xyz_locs[alpha_mask]
            sdfs = self.sdf_inter_fun(xyz_sampled)[..., 0]
            near_surf_mask = torch.abs(sdfs) < self.cfg['mul_length'] * length
            inv_s = self.deviation_network(xyz_sampled).clip(1e-6, 1e6)
            inv_s = inv_s[..., 0]
            estimated_next_sdf = sdfs - length * 0.5
            estimated_prev_sdf = sdfs + length * 0.5

            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)       # [N_rays, ]
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha_weights = ((p + 1e-5) / (c + 1e-5)).clip(min=0.0, max=1.0)
            alpha_weights[near_surf_mask] = 1
            alpha[alpha_mask] = alpha_weights
        return alpha      

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'sdf_n_comp': self.sdf_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'sdf_dim': self.sdf_dim,
            'app_dim': self.app_dim,

            'alphaMask_thres': self.alphaMask_thres,
            'marched_weights_thres' : self.marched_weights_thres,
            'step_ratio': self.step_ratio,
            'max_levels': self.max_levels,
        }

    def update_learning_rate(self, step):
        progress = step / self.cfg['lr_decay_iters']
        cur_lr_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - self.cfg['lr_decay_target_ratio']) + self.cfg['lr_decay_target_ratio']
        self.lr_factor = cur_lr_factor / self.pre_lr_factor
        self.pre_lr_factor = cur_lr_factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.lr_factor
    
    def ckpt_to_save(self):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'network_state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        ckpt['optimizer_state_dict'] = self.optimizer.state_dict()
        return ckpt

    def load_ckpt(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['network_state_dict'])
        if 'optimizer_state_dict' in ckpt.keys():
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    def load_iter(self, model_path, iteration, finetuned=False):
        if finetuned:
            load_dir = os.path.join(model_path, 'finetuned', 'sdf_render', 'iteration_' + str(iteration))
        else:
            load_dir = os.path.join(model_path, 'sdf_render', 'iteration_' + str(iteration))
        ckpt = torch.load(f'{load_dir}/tensosdf.th')
        self.load_sdf_pretrain(ckpt)
    
    def load_sdf_pretrain(self, ckpt):
        sdf_ckpt = {}
        # if 'alphaMask.aabb' in ckpt.keys():
        #     length = np.prod(ckpt['alphaMask.shape'])
        #     alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
        #     self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
        ckpt = ckpt['network_state_dict']
        for k in ckpt.keys():
            if k.startswith("sdf_network"):
                sdf_ckpt[k[len('sdf_network.'):]] = ckpt[k]
            elif k.startswith("deviation_network"):
                self.deviation_network.variance = torch.nn.Parameter(ckpt[k])
        self.sdf_network.load_state_dict(sdf_ckpt)

    def save(self, model_path, iteration):
        save_dir = os.path.join(model_path, 'sdf_render', 'iteration_' + str(iteration))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.ckpt_to_save(), f'{save_dir}/tensosdf.th')


    def upsample_sdf_grid(self, res_target):
        res_target = torch.tensor(res_target, device=self.device)
        new_res, max_levels = self.sdf_network.upsample_volume_grid(res_target)
        self.update_stepSize(new_res, max_levels)
        
    def update_aabb(self, aabb):
        self.aabb = aabb
        self.sdf_network.update_gridSize_aabb(self.gridSize, aabb, self.sdf_network.n_levels)
        self.update_stepSize(self.gridSize, self.sdf_network.n_levels)
        
        
    def shrink_sdf_grid(self, new_aabb):
        raise NotImplementedError

    def get_train_opt_params(self, learning_rate_xyz, learning_rate_net):
        grad_vars = []
        get_grad_vars_from_net = lambda net : [{'params' : net.parameters(), 'lr' : learning_rate_net}]
        grad_vars += self.sdf_network.get_optparam_groups(learning_rate_xyz, learning_rate_net)
        grad_vars += get_grad_vars_from_net(self.deviation_network)
        return grad_vars
        
    def get_human_coordinate_poses(self, poses):
        pn = poses.shape[0]
        cam_cen = (-poses[:, :, :3].permute(0, 2, 1) @ poses[:, :, 3:])[..., 0]  # pn,3
        if self.cfg['fixed_camera']:
            pass
        else:
            cam_cen[..., 2] = 0

        Y = torch.zeros([1, 3], device=poses.device).expand(pn, 3)
        Y[:, 2] = -1.0
        Z = torch.clone(poses[:, 2, :3]).to(poses.device)  # pn, 3
        Z[:, 2] = 0
        Z = F.normalize(Z, dim=-1)
        X = torch.cross(Y, Z)  # pn, 3
        R = torch.stack([X, Y, Z], 1)  # pn,3,3
        t = -R @ cam_cen[:, :, None]  # pn,3,1
        return torch.cat([R, t], -1)

    @torch.no_grad()
    def filtering_train_rays(self, device='cuda', chunk=10240*5):
        print('========> filtering rays ...')
        tt = time.time()
        rays_o, rays_d = self.train_batch['rays_o'], self.train_batch['dirs']  
        N = torch.tensor(rays_o.shape[:-1]).cpu().prod()
        aabb = self.aabb.to(device)

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)        
        for idx_chunk in idx_chunks:
            rays_o_chunk, rays_d_chunk = rays_o[idx_chunk].to(device), rays_d[idx_chunk].to(device)

            vec = torch.where(rays_d_chunk == 0, torch.full_like(rays_d_chunk, 1e-6), rays_d_chunk)
            rate_a = (aabb[1] - rays_o_chunk) / vec
            rate_b = (aabb[0] - rays_o_chunk) / vec
            t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
            t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
            mask_inbbox = t_max > t_min
            
            mask_filtered.append(mask_inbbox.cpu())
            
        mask_filtered = torch.cat(mask_filtered).view(rays_o.shape[:-1])
        valid_rn = torch.sum(mask_filtered)
        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {valid_rn / N}')
        
        for k, v in self.train_batch.items():
            self.train_batch[k] = v[mask_filtered]
        self.tbn = valid_rn

    def get_anneal_val(self, step):
        if self.cfg['anneal_end'] < 0:
            return 1.0
        else:
            return np.min([1.0, step / self.cfg['anneal_end']])

    def near_far_from_sphere(self, rays_o, rays_d):
        radius = self.radius if self.radius is not None else 1.0
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - radius
        far = mid + radius
        near = torch.clamp(near, min=1e-3)
        return near, far

    def compute_sample_level(self, pts):
        level = torch.zeros(pts.shape[:-1] + (1, ))
        return level

    def _process_ray_batch(self, ray_batch, poses):
        rays_d = ray_batch['dirs']  # rn,3
        idxs = ray_batch['idxs'][..., 0]  # rn

        rays_o = poses[:, :, :3].permute(0, 2, 1) @ -poses[:, :, 3:]  # trn,3,1
        rays_o = rays_o[idxs, :, 0]  # rn,3
        rays_d = poses[idxs, :, :3].permute(0, 2, 1) @ rays_d.unsqueeze(-1)
        rays_d = rays_d[..., 0]  # rn,3

        rays_o = rays_o
        rays_d = F.normalize(rays_d, dim=-1)
        near, far = self.near_far_from_sphere(rays_o, rays_d)

        ray_batch['rays_o'] = rays_o
        ray_batch['dirs'] = rays_d
        return ray_batch, near, far  # rn, 3, 4
    
    def _process_ray_batch_nerf(self, ray_batch, poses):
        rays_d = ray_batch['dirs']  # rn,3
        idxs = ray_batch['idxs'][..., 0]  # rn

        rays_o = poses[idxs, :3, -1] # rn,3
        rays_d = torch.sum(rays_d[..., None, :] * poses[idxs, :3, :3], -1)  # rn,3
        rays_d = F.normalize(rays_d, dim=-1)
        near, far = self.near_far_from_sphere(rays_o, rays_d)

        ray_batch['rays_o'] = rays_o
        ray_batch['dirs'] = rays_d
        return ray_batch, near, far # rn, 3, 4

    def test_step(self, data, step):
        rays_o, rays_d = data['rays_o'], data['dirs']
        near, far = self.near_far_from_sphere(rays_o, rays_d)
        outputs = self.render(data, near, far, None, 0, 0, is_train=False, step=step)

        self.zero_grad()
        return outputs

    def train_step(self, step, train_ray_batch):
        # fetch to gpu
        rays_o, rays_d = train_ray_batch['rays_o'], train_ray_batch['dirs']
        near, far = self.near_far_from_sphere(rays_o, rays_d)
        outputs = self.render(train_ray_batch, near, far, None, -1, 
                              self.get_anneal_val(step), is_train=True, step=step)
        # outputs['loss_rgb'] = self.compute_rgb_loss(outputs['ray_rgb'], train_ray_batch['rgbs'])  # ray_loss
        if train_ray_batch.get('depth') is not None:
            outputs['loss_depth_g2sdf'] = F.l1_loss(outputs['depth'], train_ray_batch['depth'].detach())       
            outputs['loss_depth_sdf2g'] = F.l1_loss(outputs['depth'].detach(), train_ray_batch['depth'])       
        # outputs['loss_normal_g2sdf'] = F.l1_loss(outputs['normal'], train_ray_batch['normal'].detach())       
        outputs['loss_normal_g2sdf'] = (1 - (outputs['normal'] * train_ray_batch['normal'].detach()).sum(-1)).mean()
        # outputs['loss_normal_sdf2g'] = F.l1_loss(outputs['normal'].detach(), train_ray_batch['normal'])  
        outputs['loss_normal_sdf2g'] =  (1 - (outputs['normal'].detach() * train_ray_batch['normal']).sum(-1)).mean()
        if self.cfg['apply_mask_loss']:
            outputs['loss_mask'] = F.binary_cross_entropy(outputs['acc'].clip(1e-4, 1.0 - 1e-4), (train_ray_batch['masks'] > 0.5).float())
        loss_reg_sdf = self.sdf_init_loss(outputs, step)
        outputs.update(loss_reg_sdf)
        return outputs

    def compute_rgb_loss(self, rgb_pr, rgb_gt):
        if self.cfg['rgb_loss'] == 'l2':
            rgb_loss = torch.sum((rgb_pr - rgb_gt) ** 2, -1)
        elif self.cfg['rgb_loss'] == 'l1':
            rgb_loss = torch.sum(F.l1_loss(rgb_pr, rgb_gt, reduction='none'), -1)
        elif self.cfg['rgb_loss'] == 'smooth_l1':
            rgb_loss = torch.sum(F.smooth_l1_loss(rgb_pr, rgb_gt, reduction='none', beta=0.25), -1)
        elif self.cfg['rgb_loss'] == 'charbonier':
            epsilon = 0.001
            rgb_loss = torch.sqrt(torch.sum((rgb_gt - rgb_pr) ** 2, dim=-1) + epsilon)
        else:
            raise NotImplementedError
        return rgb_loss

    def density_activation(self, density, dists):
        return 1.0 - torch.exp(-F.softplus(density) * dists)

    def compute_density(self, points):
        points_norm = torch.norm(points, dim=-1, keepdim=True)
        points_norm = torch.clamp(points_norm, min=1e-3)
        sigma = self.outer_nerf.density(torch.cat([points / points_norm, 1.0 / points_norm], -1))[..., 0]
        return sigma

    @staticmethod
    def upsample(rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        level = self.compute_sample_level(pts) # [rn, sn, 1]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)
        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3), level).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def sample_ray(self, rays_o, rays_d, near, far, perturb):
        n_samples = self.cfg['n_samples'] 
        n_bg_samples = self.cfg['n_bg_samples']
        n_importance = self.cfg['n_importance']
        up_sample_steps = self.cfg['up_sample_steps']

        # sample points
        batch_size = len(rays_o)
        z_vals = torch.linspace(0.0, 1.0, n_samples)  # sn
        
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near[..., 0], max=far[..., 0]).unsqueeze(-1)
        t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near[..., 0], max=far[..., 0]).unsqueeze(-1)
        
        z_vals = t_min + (t_max - t_min) * z_vals[None, :]  # rn,sn
        if n_bg_samples > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (n_bg_samples + 1.0), n_bg_samples)

        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / n_samples

            if n_bg_samples > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        # Up sample
        with torch.no_grad():
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None] # [rn, sn, 3]
            level = self.compute_sample_level(pts) # [rn, sn, 1]
            sdf = self.sdf_network.sdf(pts, level).reshape(batch_size, n_samples)

            for i in range(up_sample_steps):
                rn, sn = z_vals.shape
                if self.cfg['clip_sample_variance']:
                    inv_s = self.deviation_network(torch.empty([1, 3])).expand(rn, sn - 1)
                    inv_s = torch.clamp(inv_s, max=64 * 2 ** i)  # prevent too large inv_s
                else:
                    inv_s = torch.ones(rn, sn - 1) * 64 * 2 ** i
                new_z_vals = self.upsample(rays_o, rays_d, z_vals, sdf, n_importance // up_sample_steps, inv_s)
                z_vals, sdf = self.cat_z_vals(rays_o, rays_d, z_vals, new_z_vals, sdf, last=(i + 1 == up_sample_steps))

        if n_bg_samples > 0:
            z_vals = torch.cat([z_vals, z_vals_outside], -1)
        return z_vals
    
    def sample_ray_depth(self, rays_o, rays_d, near, far, perturb, pos, depth, acc):
        n_samples = 64
        n_importance = 64
        up_sample_steps = 4

        sdf = self.sdf_inter_fun(pos).detach()

        # sample points
        batch_size = len(rays_o)
        # z_vals = depth + torch.linspace(-1.0 * sdf, 1.0 * sdf, n_samples)  # sn
        z_vals = torch.linspace(0.0, 1.0, n_samples)  # sn
        
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near[..., 0], max=far[..., 0]).unsqueeze(-1)
        t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near[..., 0], max=far[..., 0]).unsqueeze(-1)
        z_vals = t_min + (t_max - t_min) * z_vals[None, :]  # rn,sn
        mask = torch.logical_and(sdf < 0.2, depth>near).squeeze()
        if mask.any():
            z_sdf = depth[mask] + torch.linspace(-1.0, 1.0, n_samples)[None, :] * 4 * torch.abs(sdf[mask])
            z_vals[mask] = z_sdf.clamp(near[mask])

        # if perturb > 0:
        #     t_rand = (torch.rand([batch_size, 1]) - 0.5)
        #     z_vals = z_vals + t_rand * 2.0 / n_samples

        #     if n_bg_samples > 0:
        #         mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
        #         upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
        #         lower = torch.cat([z_vals_outside[..., :1], mids], -1)
        #         t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
        #         z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        # if n_bg_samples > 0:
        #     z_vals_outside = t_max / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / n_bg_samples

        # Up sample
        with torch.no_grad():
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None] # [rn, sn, 3]
            level = self.compute_sample_level(pts) # [rn, sn, 1]
            sdf = self.sdf_network.sdf(pts, level).reshape(batch_size, n_samples)

            for i in range(up_sample_steps):
                rn, sn = z_vals.shape
                if self.cfg['clip_sample_variance']:
                    inv_s = self.deviation_network(torch.empty([1, 3])).expand(rn, sn - 1)
                    inv_s = torch.clamp(inv_s, max=64 * 2 ** i)  # prevent too large inv_s
                else:
                    inv_s = torch.ones(rn, sn - 1) * 64 * 2 ** i
                new_z_vals = self.upsample(rays_o, rays_d, z_vals, sdf, n_importance // up_sample_steps, inv_s)
                z_vals, sdf = self.cat_z_vals(rays_o, rays_d, z_vals, new_z_vals, sdf, last=(i + 1 == up_sample_steps))

        # if n_bg_samples > 0:
        #     z_vals = torch.cat([z_vals, z_vals_outside], -1)
        return z_vals

    def render(self, ray_batch, near, far, human_poses, perturb_overwrite=-1, 
               cos_anneal_ratio=0.0, is_train=True, step=None, z_vals=None):
        """
        :param ray_batch: rn,x
        :param near:   rn,1
        :param far:    rn,1
        :param human_poses:     rn,3,4
        :param perturb_overwrite: set 0 for inference
        :param cos_anneal_ratio:
        :param is_train:
        :param step:
        :return:
        """
        perturb = self.cfg['perturb']
        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        rays_o, rays_d = ray_batch['rays_o'], ray_batch['dirs']
        depths = ray_batch.get('depth', None)
        pos = ray_batch.get('pos', None)
        bg = ray_batch.get('bg', 1)
        acc = ray_batch.get('acc')
        z_vals = self.sample_ray(rays_o, rays_d, near, far, perturb)
        ret = self.render_core(rays_o, rays_d, z_vals, human_poses, 
                               cos_anneal_ratio=cos_anneal_ratio, step=step, 
                               is_train=is_train, bg=bg)
        return ret

        
    def compute_sdf_alpha(self, points, level, dists, dirs, cos_anneal_ratio, step, is_train):
        # points [...,3] dists [...] dirs[...,3]
        sdf_nn_output = self.sdf_network(points, level)
        sdf = sdf_nn_output[..., 0]
        feature_vector = sdf_nn_output[..., 1:]

        gradients, hessian = self.sdf_network.gradient(points, level, training=is_train, sdf=sdf[..., None])  # ...,3
        inv_s = self.deviation_network(points).clip(1e-6, 1e6)  # ...,1
        inv_s = inv_s[..., 0]

        if self.cfg['freeze_inv_s_step'] is not None and step < self.cfg['freeze_inv_s_step']:
            inv_s = inv_s.detach()

        true_cos = (dirs * gradients).sum(-1)  # [...]
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)  # [...]
        return alpha, gradients, feature_vector, inv_s, sdf, hessian

    def compute_density_alpha(self, points, dists, dirs, nerf):
        norm = torch.norm(points, dim=-1, keepdim=True)
        points = torch.cat([points / norm, 1.0 / norm], -1)
        density, color = nerf(points, dirs)  # [...,1] [...,3]
        alpha = self.density_activation(density[..., 0], dists)
        color = linear_to_srgb(torch.exp(torch.clamp(color, max=5.0)))
        return alpha, color

    def render_core(self, rays_o, rays_d, z_vals, human_poses=None, 
                    cos_anneal_ratio=0.0, step=None, is_train=True,
                    bg=None):
        batch_size, n_samples = z_vals.shape

        # section length in original space
        dists = z_vals[..., 1:] - z_vals[..., :-1]  # rn,sn-1
        dists = torch.cat([dists, dists[..., -1:]], -1)  # rn,sn
        mid_z_vals = z_vals + dists * 0.5
        
        points = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * mid_z_vals.unsqueeze(-1) # [rn, sn, 3]
        level = self.compute_sample_level(points) # [rn, sn, 1]
        derived_normals = torch.zeros(batch_size, n_samples, 3)

        outer_mask = ((self.aabb[0]>points) | (points>self.aabb[1])).any(dim=-1)
        inner_mask = ~outer_mask      

        dirs = rays_d.unsqueeze(-2).expand(batch_size, n_samples, 3)
        dirs = F.normalize(dirs, dim=-1)
        alpha, sampled_color = torch.zeros(batch_size, n_samples), torch.zeros(batch_size, n_samples, 3)
        
        alpha_rest_ratio = 1.0
        if self.alphaMask is not None:
            alpha_mask = self.alphaMask.sample_alpha(points[inner_mask]) > 0
            alpha_rest_ratio = 1.0 - torch.sum(~alpha_mask) / torch.sum(inner_mask)
            inner_mask_invalid = ~inner_mask
            inner_mask_invalid[inner_mask] |= (~alpha_mask)
            inner_mask = ~inner_mask_invalid 
                
        if torch.sum(inner_mask) > 0:   
            # add GPU Memory             
            alpha[inner_mask], gradients, feature_vector, inv_s, sdf, hessian = self.compute_sdf_alpha(
                points[inner_mask], level[inner_mask], dists[inner_mask], dirs[inner_mask], cos_anneal_ratio, step, is_train)
            derived_normals[inner_mask] = gradients
            gradient_error = (torch.linalg.norm(gradients, ord=2, dim=-1) - 1.0) ** 2
            
            if self.cfg['apply_sparse_loss']:
                gamma = 20.
                reg_loss = torch.exp(-gamma * sdf.abs())        # [..., ]
                reg_loss = reg_loss.sum() / (inner_mask.sum() + 1e-5) * alpha_rest_ratio
                
            if self.cfg['apply_hessian_loss'] and hessian is not None:
                hessian_loss = hessian.abs().sum() / (inner_mask.sum() + 1e-5) * alpha_rest_ratio
            else:
                hessian_loss = torch.zeros(1)
        else:
            gradient_error = torch.zeros(1)
            if self.cfg['apply_sparse_loss']:
                reg_loss = torch.zeros(1)
            if self.cfg['apply_hessian_loss']:
                hessian_loss = torch.zeros(1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 
                                                   1. - alpha + 1e-7], -1), -1)[..., :-1]  # rn,sn
        acc_map = torch.sum(weights, dim=-1, keepdim=True) # rn, 1          
        if not self.cfg['predict_BG']:
            color = (sampled_color * weights[..., None]).sum(dim=1) + (1 - acc_map) * bg
        else:
            color = (sampled_color * weights[..., None]).sum(dim=1)
        outputs = {
            'ray_rgb': color,  # rn,3
            'loss_eikonal': gradient_error.mean(),  # rn
            'acc' : acc_map,
        }

        acc_sampled_normal = (derived_normals * weights[..., None]).sum(dim=1)
        depth = (weights[..., None] * z_vals[..., None]).sum(dim=1)
        outputs['normal'] = F.normalize(acc_sampled_normal * acc_map + (1. - acc_map) * torch.tensor([0.0, 0.0, 1.0], device=acc_sampled_normal.device), dim=-1)
        outputs['depth'] = depth

        if torch.sum(inner_mask) > 0:
            outputs['std'] = torch.mean(1 / inv_s)
        else:
            outputs['std'] = torch.zeros(1)

        if step < 1000:
            if torch.sum(inner_mask) > 0:
                outputs['sdf_pts'] = points[inner_mask]
                outputs['sdf_vals'] = self.sdf_network.sdf(points[inner_mask], level[inner_mask])[..., 0]
            else:
                outputs['sdf_pts'] = torch.zeros(1)
                outputs['sdf_vals'] = torch.zeros(1)

        if self.cfg['apply_gaussian_loss'] and step > self.cfg['gaussianLoss_step']:
            # gaussian loss
            if torch.sum(inner_mask) > 0:
                outputs['loss_gaussian'] = self.sdf_network.grid_gaussian_loss()
            else:
                outputs['loss_gaussian'] = torch.zeros(1)

        if self.cfg['apply_tv_loss']:
            outputs['loss_tv_sdf'] = self.sdf_network.TV_loss_sdf(self.tv_reg)          

        if self.cfg['apply_sparse_loss']:
            outputs['loss_sparse'] = reg_loss

        if self.cfg['apply_hessian_loss']:
            outputs['loss_hessian'] = hessian_loss

        return outputs

    def forward(self, data, is_train=True):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        step = data['step']
        if is_train:
            outputs = self.train_step(step, data)
        else:
            outputs = self.test_step(data, step)
        torch.set_default_tensor_type('torch.FloatTensor')
        return outputs

    def filter_gaussians(self, means3D):
        outer_mask = ((self.aabb[0]>means3D) | (means3D>self.aabb[1])).any(dim=-1)
        inner_mask = ~outer_mask      

        if inner_mask.any():
            with torch.no_grad():
                sdf = self.sdf_inter_fun(means3D[inner_mask])
                inv_s = self.deviation_network(means3D[inner_mask]).clone().detach()
                s = (1/inv_s).clamp(1e-6, 1e6)
                a = 1e-2
                exp_r = s/2/a - 1 - torch.sqrt(s*(s - 4*a))/2/a
                threshold = (-torch.log(exp_r) / s).clamp(1e-4, 0.05)
                threshold = 0.03

            sdf_mask = torch.abs(sdf) > threshold
            
        outer_mask[inner_mask] = sdf_mask.squeeze()
        
        return outer_mask
    
