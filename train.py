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

import os
import torch
import imageio
import torch.nn.functional as F
from random import randint
from utils.loss_utils import (l1_loss, l2_loss, ssim, predicted_normal_loss, delta_normal_loss, 
                              zero_one_loss, bilateral_smooth_loss, base_smooth_loss, depth_smooth_loss)
from gaussian_renderer import RENDER_DICT, render_lighting
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import apply_depth_colormap
from utils.graphics_utils import depths_to_points
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, TensoSDFOptimParams
import torchvision
from time import time
from fields.shape_renders import SDF_RENDER_DICT
from scene.NVDIFFREC.light import load_latlong_env
from fused_ssim import fused_ssim 

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, sdf_opt, testing_iterations, saving_iterations, load_iteration):
    render_fn = RENDER_DICT[pipe.gaussian_type]
    tb_writer = prepare_output_and_logger(dataset, opt, pipe)
    gaussians = GaussianModel(dataset.sh_degree, dataset.env_mode, dataset.envmap_res,
                              dataset.use_delta, opt.metallic)
    sdf_render = SDF_RENDER_DICT[sdf_opt.sdf_mode]({}).cuda()
    sdf_render.training_setup(sdf_opt)
    if load_iteration > -1:
        scene = Scene(dataset, gaussians, load_iteration=load_iteration)
        # sdf_render.load_iter(scene.model_path, load_iteration)
    else:
        scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt, scene)
    print("decay:", sdf_opt.decay)
    sdf_ckpt = None
    sdf_pretrained = False
    if os.path.exists(sdf_opt.pretrain_sdf):
        sdf_ckpt = torch.load(sdf_opt.pretrain_sdf)
        sdf_render.load_sdf_pretrain(sdf_ckpt)
        sdf_render.freeze_sdf()
        sdf_pretrained = True
        print("**********************************************")
        print('load sdf pretrained model from', sdf_ckpt)
        print("**********************************************")
    if len(pipe.envmap_path) > 0:
        print('load pretrained envmap...')
        if gaussians.env_mode.find('mc') > -1:
            envmap = load_latlong_env(pipe.envmap_path).cuda()
            trans_mat = torch.tensor([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).cuda().float()
            gaussians.brdf_mlp.set_base(envmap, trans_mat)

        else:
            raise NotImplementedError

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    load_iteration = load_iteration if load_iteration > 0 else 0
    progress_bar = tqdm(range(load_iteration, opt.iterations), desc="Training progress")
     

    init_flag = True
    # gaussians.compute_3D_filter(scene.getTrainCameras())
    for iteration in range(load_iteration+1, opt.iterations + 1): 
        if dataset.random_background == True:
            background = torch.rand((3)).cuda()

        iter_start = time()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) 
        
        mask = viewpoint_cam.gt_alpha_mask.cuda()
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        
        gt_image = viewpoint_cam.original_image.cuda()
        

        gaussians.set_requires_grad("normal", state=iteration >= opt.normal_reg_from_iter)
        gaussians.set_requires_grad("normal2", state=iteration >= opt.normal_reg_from_iter)
        if gaussians.env_mode=="envmap":
            gaussians.brdf_mlp.build_mips()

        # Render
        rad_only = opt.pbr_from_iter >= iteration
        render_pkg = render_fn(viewpoint_cam, scene, pipe, background, 
                            debug=False, is_training=True)
        image, image_rad = render_pkg["render"], render_pkg.get("render_rad")
        # print(gaussians.get_roughness.max(), gaussians.get_roughness.min())
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        sdf_losses = {}
        gt_image = gt_image * mask + background[:, None, None] * (1-mask)
        loss = torch.tensor(0.).cuda()
        if iteration > sdf_opt.sdf_from_iter:
            sdf_render.train()
            if sdf_opt.sdf_init_iters > 0 and init_flag:
                sdf_init(scene.getTrainCameras().copy(), sdf_opt, pipe, scene, render_fn,
                        sdf_render, pretrained_iters=sdf_opt.sdf_init_iters)
                init_flag = False
            pos = render_pkg['pos'].permute(1, 2, 0)
            depth = render_pkg['depth'].permute(1, 2, 0)
            normal = render_pkg['normal'].permute(1, 2, 0)
            acc = render_pkg['alpha'].permute(1, 2, 0)

            viewdirs, valid_mask = viewpoint_cam.get_filtered_ray()
            valid_normal = normal.view(-1, 3)
            valid_depth = depth.view(-1, 1)
            valid_viewdirs = viewdirs.view(-1, 3)
            valid_pos = pos.view(-1, 3)
            valid_acc = acc.view(-1, 1)
            bs = valid_viewdirs.shape[0]
            valid_gt = gt_image.permute(1, 2, 0).view(-1, 3)
            if sdf_opt.batchify and bs > sdf_opt.batch_size:
                idx = torch.randint(bs, [sdf_opt.batch_size])
                valid_gt = valid_gt[idx]
                valid_viewdirs = valid_viewdirs[idx]
                valid_normal = valid_normal[idx]
                valid_depth = valid_depth[idx]
                valid_pos = valid_pos[idx]
                valid_mask = mask.permute(1, 2, 0).view(-1, 1)[idx]
                valid_acc = valid_acc[idx]
                default_normal = torch.tensor([[0, 0, 1]]).float().cuda()
                valid_normal = valid_normal * valid_acc + (1-valid_acc) * default_normal
                valid_normal = F.normalize(valid_normal, dim=-1) 
                bs = sdf_opt.batch_size
            ray_batch = {
                'rays_o': viewpoint_cam.camera_center.repeat(bs, 1),
                'rgbs': valid_gt,
                'dirs': valid_viewdirs, 
                'step': iteration + sdf_opt.sdf_init_iters + 999999,
                'masks': valid_mask,
                'bg': background[None, :],
                'depth': valid_depth,
                'pos': valid_pos,
                'normal': valid_normal,
                "acc": valid_acc,
                'pretrained': sdf_pretrained,
            }
            tensosdf_output = sdf_render(ray_batch,)
            for key in tensosdf_output.keys():
                if key.find('loss') > -1:
                    sdf_loss = tensosdf_output[key].mean()* getattr(sdf_opt, f'lambda_'+key[5:])
                    sdf_losses[key] = sdf_loss 
                    loss += sdf_loss
        if opt.pbr_from_iter < iteration:
            Ll1 = l1_loss(image, gt_image)
            loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
        losses_extra = {}
        
        if iteration > opt.normal_reg_from_iter and iteration < opt.normal_reg_util_iter:
            if iteration < opt.normal_reg_util_iter and "normal_ref" in render_pkg.keys():
                losses_extra['predicted_normal'] = predicted_normal_loss(render_pkg["normal"], 
                                                render_pkg["normal_ref"], mask)
            if opt.zero_one_use_gt:
                o = render_pkg["alpha"].clamp(1e-6, 1-1e-6)
                losses_extra['zero_one'] = -(mask * torch.log(o) + (1 - mask) * torch.log(1 - o)).mean()
            else:
                losses_extra['zero_one'] = zero_one_loss(render_pkg["alpha"])
            if "delta_normal_norm" in render_pkg.keys():
                losses_extra['delta_reg'] = delta_normal_loss(render_pkg["delta_normal_norm"], mask)
        
        if opt.lambda_brdf_smoothness > 0 and not rad_only:
            l_diff = bilateral_smooth_loss(render_pkg['diffuse'], gt_image, mask) if 'diffuse' in render_pkg.keys() else 0
            l_spec = bilateral_smooth_loss(render_pkg['specular'], gt_image, mask) if 'specular' in render_pkg.keys() else 0
            l_roughness = base_smooth_loss(render_pkg['roughness'], gt_image, mask)
            l_albedo = bilateral_smooth_loss(render_pkg['albedo'], gt_image, mask)
            l_metallic = base_smooth_loss(render_pkg['metallic'], gt_image, mask)
            losses_extra['brdf_smoothness'] = l_diff + l_spec + l_roughness + l_albedo + l_metallic

        if opt.lambda_base_smoothness > 0:
            l_roughness = base_smooth_loss(render_pkg['roughness'], gt_image, mask) 
            l_roughness += (render_pkg['roughness'] - 0.5).clamp(0, 1).mean() * 0.25
            l_albedo = base_smooth_loss(render_pkg['albedo'], gt_image, mask)
            l_metallic = base_smooth_loss(render_pkg['metallic'], gt_image, mask) 
            losses_extra['base_smoothness'] = l_roughness + l_albedo + l_metallic
            

        if opt.lambda_opacity_reg > 0 and iteration > opt.opacity_01_from_iter and iteration < opt.opacity_01_under_iter:
                opacities = gaussians.get_opacity
                losses_extra['opacity_reg'] =  (
                        - opacities * torch.log(opacities + 1e-10)
                        - (1 - opacities) * torch.log(1 - opacities + 1e-10)
                        ).mean()
        
        if opt.lambda_light_reg > 0.:
            diffuse_lights = render_pkg.get('diffuse_light')
            if diffuse_lights is not None and pipe.env_mode=="fnero":
                losses_extra['light_reg'] = torch.mean(torch.abs(diffuse_lights - torch.mean(diffuse_lights, dim=-1, keepdim=True)))
            else:
                losses_extra['light_reg'] = gaussians.brdf_mlp.regularizer()
        
        for k in losses_extra.keys():
            loss += getattr(opt, f'lambda_{k}')* losses_extra[k]
        loss.backward()
        iter_end = time()
        scene.gaussians._roughness.grad[scene.gaussians._roughness.grad.isnan()] = 0.0

        # with torch.enable_grad():
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 1 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{3}f}, {gaussians.get_xyz.shape[0]}"})
                progress_bar.update(1)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], 
                                                                radii[visibility_filter])

            # Log and save
            losses_extra['psnr'] = psnr(image, gt_image).mean()
            losses_extra.update(sdf_losses)
            test_bg = torch.tensor([1,1,1], dtype=torch.float32, device="cuda")
            training_report(tb_writer, iteration, Ll1, loss, losses_extra, l1_loss, iter_end - iter_start, 
                            testing_iterations, scene, render_fn, sdf_render, (pipe, test_bg), 
                            {"is_training": False},
                            {"batchify": sdf_opt.batchify, "batch_size": 1536, 
                            "pretrained": sdf_pretrained})
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                sdf_render.save(scene.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005+1000, opt.opacity_cull,
                                                scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration > opt.opacity_remove_from_iter and iteration % 1000 == 1: 
                opacity = gaussians.get_opacity
                prune_mask = (opacity < opt.opacity_remove_threshold).squeeze()
                gaussians.prune_points(prune_mask)
                print(f'[Iter {iteration}]: prune low opacity gaussians, pruning number: {int(prune_mask.sum())}')
                torch.cuda.empty_cache()
                
            if iteration > opt.sdf_remove_from_iter and iteration % 1000 == 1:
                outer_mask = sdf_render.filter_gaussians(gaussians.get_xyz)
                print(f'prune {outer_mask.sum()}')
                gaussians.prune_points(outer_mask)
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                sdf_render.optimizer.step()
                gaussians.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)
                sdf_render.optimizer.zero_grad(set_to_none=True)

            if sdf_opt.sdf_mode.find("Tenso") > -1 and sdf_opt.update_alpha_mask_list is not None and iteration in sdf_opt.update_alpha_mask_list:
                new_aabb = sdf_render.updateAlphaMask()
                print('new_aabb: ', new_aabb)
            
            if sdf_opt.sdf_mode.find("Tenso") > -1 and sdf_opt.upsample_list is not None and iteration in sdf_opt.upsample_list:
                # upsamp_gridSize = 2 * self.network.gridSize
                n_voxels = sdf_render.N_voxel_list.pop(0)
                upsamp_gridSize = sdf_render.N_to_reso(n_voxels, sdf_render.aabb)
                sdf_render.upsample_sdf_grid(upsamp_gridSize)
                
                sdf_render.training_setup(sdf_opt, sdf_opt.decay)
            if sdf_opt.decay:
                sdf_render.update_learning_rate(iteration)
            if pipe.env_mode=="envmap" and pipe.render_mode.find('split_sum') < 0 and pipe.render_mode.find('disney') < 0:
                gaussians.brdf_mlp.clamp_(min=0.0, max=1.0)

def sdf_init(train_cams, sdf_opt, pipe, scene, render_fn, sdf_render, pretrained_iters=1000):
    sdf_losses = {}
    background = torch.rand((3)).cuda()
    batch_size = 2048
    gs_pos = scene.gaussians.get_xyz
    min_v = torch.min(gs_pos.min(0)[0]*1.2, -torch.ones_like(gs_pos.min(0)[0])*1.5).detach().clone()
    max_v = torch.max(gs_pos.max(0)[0]*1.2, torch.ones_like(gs_pos.max(0)[0])*1.5).detach().clone()
    aabb = torch.cat([min_v, max_v], 0).view(2, 3)
    sdf_render.update_aabb(aabb)
    for i in tqdm(range(pretrained_iters)):
        viewpoint_cam = train_cams[randint(0, len(train_cams)-1)]
        gt_image = viewpoint_cam.original_image.cuda()
        mask = viewpoint_cam.gt_alpha_mask.cuda()
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        gt_image = gt_image * mask + background[:, None, None] * (1-mask)
        render_pkg = render_fn(viewpoint_cam, scene, pipe, background, 
                            debug=False, is_training=True, attr_only=True)
        pos = render_pkg['pos'].permute(1, 2, 0)
        depth = render_pkg['depth'].permute(1, 2, 0).detach()
        normal = render_pkg['normal'].permute(1, 2, 0).detach()
        acc = render_pkg['alpha'].permute(1, 2, 0)

        viewdirs, valid_mask = viewpoint_cam.get_filtered_ray()
        valid_normal = normal.view(-1, 3)
        valid_depth = depth.view(-1, 1)
        valid_viewdirs = viewdirs.view(-1, 3)
        valid_pos = pos.view(-1, 3)
        valid_acc = acc.view(-1, 1)
        bs = valid_viewdirs.shape[0]
        valid_gt = gt_image.permute(1, 2, 0).view(-1, 3)
        if  bs > batch_size:
            idx = torch.randint(bs, [batch_size])
            valid_gt = valid_gt[idx]
            valid_viewdirs = valid_viewdirs[idx]
            valid_normal = valid_normal[idx]
            valid_depth = valid_depth[idx]
            valid_pos = valid_pos[idx]
            valid_mask = mask.permute(1, 2, 0).view(-1, 1)[idx]
            valid_acc = valid_acc[idx]
            default_normal = torch.tensor([[0, 0, 1]]).float().cuda()
            valid_normal = valid_normal * valid_acc + (1-valid_acc) * default_normal
            valid_normal = F.normalize(valid_normal, dim=-1) 
            bs = batch_size
        ray_batch = {
            'rays_o': viewpoint_cam.camera_center.repeat(bs, 1),
            'rgbs': valid_gt,
            'dirs': valid_viewdirs, 
            'step': i,
            'masks': valid_mask,
            'bg': background[None, :],
            'depth': valid_depth,
            'pos': valid_pos,
            'normal': valid_normal,
        }
        tensosdf_output = sdf_render(ray_batch,)
        loss = torch.tensor(0.).cuda().float()
        for key in tensosdf_output.keys():
            if key.find('sdf2g') > -1:
                continue
            if key.find('loss') > -1:
                sdf_loss = tensosdf_output[key].mean()* getattr(sdf_opt, f'lambda_'+key[5:])
                sdf_losses[key] = sdf_loss 
                loss += sdf_loss
        sdf_render.optimizer.zero_grad()
        loss.backward()
        sdf_render.optimizer.step()
    scene.gaussians.optimizer.zero_grad(set_to_none=True)    
    if sdf_opt.sdf_mode.find("Tenso") > -1:
        new_aabb = sdf_render.updateAlphaMask()
        print('new_aabb: ', new_aabb)
        
    
def prepare_output_and_logger(args, opt, pipe):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    with open(os.path.join(args.model_path, "opt_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(opt))))
    with open(os.path.join(args.model_path, "pipe_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(pipe))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def save_exr(img_tensor, path):
    img = img_tensor.permute(1, 2, 0).to("cpu", torch.float32).numpy()
    imageio.imwrite(path, img, format='EXR')

def training_report(tb_writer, iteration, Ll1, loss, losses_extra, l1_loss, elapsed, 
                    testing_iterations, scene : Scene, renderFunc, sdf_render, 
                    renderArgs, renderArgDict={}, sdfArgDict={}):
    number_gaussian = int((scene.gaussians.get_xyz).shape[0])
    if tb_writer:
        tb_writer.add_scalar('train/number_gaussian', number_gaussian, iteration)
        tb_writer.add_scalar('train/inv_deviation', scene.gaussians.inverse_deviation.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        for k in losses_extra.keys():
            tb_writer.add_scalar(f'train_loss_patches/{k}_loss', losses_extra[k].item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        sdf_render.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        with torch.no_grad():
        # with torch.enable_grad():
            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    images = torch.tensor([], device="cuda")
                    gts = torch.tensor([], device="cuda")
                    vis_idx = [i for i in range(0, len(config['cameras']))]
                    for idx, viewpoint in enumerate(config['cameras']):
                        if config['name'] == 'train' and idx not in vis_idx:
                            continue
                        mask = viewpoint.gt_alpha_mask.cuda()
                        gt_image = viewpoint.original_image.cuda()
                        H, W = gt_image.shape[1:]
                        mask[mask < 0.5] = 0
                        mask[mask >= 0.5] = 1
                        render_pkg = renderFunc(viewpoint, scene, *renderArgs, **renderArgDict)
                        pos = render_pkg['pos'].permute(1, 2, 0)
                        depth = render_pkg['depth'].permute(1, 2, 0)
                        sdf_normal = torch.zeros((*gt_image.shape[1:], 3), device='cuda', dtype=torch.float32)
                        sdf_rgb = torch.zeros((*gt_image.shape[1:], 3), device='cuda', dtype=torch.float32)
                        viewdirs = viewpoint.get_rays()
                        valid_viewdirs = viewdirs.view(-1, 3)
                        bs = valid_viewdirs.shape[0]
                        valid_gt = (gt_image * mask + 1 - mask).permute(1, 2, 0).view(-1, 3)
                        valid_mask = mask.view(-1, 1)
                        if idx % 5 == 0:
                            if sdfArgDict.get('batchify') and bs > sdfArgDict.get('batch_size'):
                                batch_size = sdfArgDict.get('batch_size')
                                chunk_idxs = torch.split(torch.arange(bs), batch_size) 
                                outputs_dict = {}
                                for chunk_idx in chunk_idxs:
                                    ray_batch = {
                                        'rays_o': viewpoint.camera_center.repeat(len(chunk_idx), 1),
                                        'rgbs': valid_gt[chunk_idx],
                                        'dirs': valid_viewdirs[chunk_idx], 
                                        'step': iteration + 99999999,
                                        'depth': depth.view(-1, 1)[chunk_idx],
                                        'pos': pos.view(-1, 3)[chunk_idx],
                                        'pretrained': sdfArgDict["pretrained"],
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
                                    if k in ['ray_rgb', 'normal', "acc", "depth"]:
                                        outputs_dict[k] = torch.cat(outputs_dict[k], dim=0)
                            else:
                                bs = valid_viewdirs.shape[0]
                                ray_batch = {
                                        'rays_o': viewpoint.camera_center.repeat(bs, 1),
                                        'rgbs': valid_gt,
                                        'dirs': valid_viewdirs, 
                                        'step': iteration + 9999999999,
                                        'depth': depth[chunk_idx],
                                        'pos': pos[chunk_idx],
                                    }
                                outputs_dict = sdf_render(ray_batch,)
                            sdf_normal = outputs_dict['normal'].detach()
                            sdf_rgb = outputs_dict['ray_rgb'].detach()
                            sdf_acc = outputs_dict['acc'].detach()
                            sdf_depth = outputs_dict['depth'].detach()
                            render_pkg['sdf_normal'] = sdf_normal.view(H, W, 3).permute(2, 0, 1)
                            render_pkg['sdf_rgb'] = sdf_rgb.detach().view(H, W, 3).permute(2, 0, 1)
                            render_pkg['sdf_acc'] = sdf_acc.detach().view(H, W, 1).permute(2, 0, 1)
                            render_pkg['sdf_depth'] = sdf_depth.detach().view(H, W, 1).permute(2, 0, 1)
                        
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        images = torch.cat((images, image.unsqueeze(0)), dim=0)
                        gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)
                        if tb_writer and (idx in vis_idx):
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            for k in render_pkg.keys():
                                hdr_format = False
                                if render_pkg[k].dim()<3 or k=="delta_normal_norm":
                                    continue
                                if "depth" in k:
                                    image_k = apply_depth_colormap(-render_pkg[k][0][...,None])
                                    image_k = image_k.permute(2,0,1)
                                elif k == "alpha":
                                    image_k = apply_depth_colormap(render_pkg[k][0][...,None], min=0., max=1.)
                                    image_k = image_k.permute(2,0,1)
                                elif k == 'render':
                                    image_k = render_pkg['render']
                                    hdr_format = renderArgs[0].hdr
                                else:
                                    if "normal" in k:
                                        render_pkg[k] = 0.5 + (0.5*render_pkg[k]) # (-1, 1) -> (0, 1)
                                    image_k = torch.clamp(render_pkg[k], 0.0, 1.0)
                                tb_writer.add_images(config['name'] + "_view_{}/{}".format(viewpoint.image_name, k), 
                                                    image_k[None], global_step=iteration)
                                save_dir = f'{scene.model_path}/eval/iteration_{iteration:05d}/{config["name"]}'
                                os.makedirs(save_dir, exist_ok=True)
                                if hdr_format:
                                    save_exr(image_k, f'{save_dir}/{idx:03d}_{k}.exr')
                                else:
                                    torchvision.utils.save_image(image_k, f'{save_dir}/{idx:03d}_{k}.png')
                            lighting = render_lighting(scene.gaussians, resolution=(512, 1024))
                            if tb_writer:
                                tb_writer.add_images(config['name'] + "/lighting", lighting[None], global_step=iteration)
                    l1_test = l1_loss(images, gts)
                    psnr_test = psnr(images, gts).mean()  
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                    if tb_writer:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    sdfop = TensoSDFOptimParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6019)
    parser.add_argument('--iteration', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000, 31_000, 36_000, 40_000, 50_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 15_000, 30_000, 40_000, 50_000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    start_time = time()
    training(lp.extract(args), op.extract(args), pp.extract(args), 
             sdfop.extract(args),
             args.test_iterations, args.save_iterations, args.iteration)
    end_time = time()
    print(end_time - start_time)
    # All done
    print("\nTraining complete.")
