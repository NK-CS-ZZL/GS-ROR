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
import math
from scene import Scene, GaussianModel
from utils.general_utils import flip_align_view
from scene.NVDIFFREC import extract_env_map
import torch.nn.functional as F
from r3dg_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def rendered_world2cam(viewpoint_cam, normal, alpha, bg_color):
    # normal: (3, H, W), alpha: (H, W), bg_color: (3)
    # normal_cam: (3, H, W)
    _, H, W = normal.shape
    _, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()
    normal_world = normal.permute(1,2,0).reshape(-1, 3) # (HxW, 3)
    normal_cam = torch.cat([normal_world, torch.ones_like(normal_world[...,0:1])], axis=-1) @ torch.inverse(torch.inverse(extrinsic_matrix).transpose(0,1))[...,:3]
    normal_cam = normal_cam.reshape(H, W, 3).permute(2,0,1) # (H, W, 3)
    
    background = bg_color[...,None,None]
    normal_cam = normal_cam*alpha[None,...] + background*(1. - alpha[None,...])

    return normal_cam

# render 360 lighting for a single gaussian
def render_lighting(pc : GaussianModel, resolution=(512, 1024)):
    if pc.env_mode=="envmap":
        lighting = extract_env_map(pc.brdf_mlp, resolution) # (H, W, 3)
        lighting = lighting.permute(2,0,1) # (3, H, W)
    else:
        raise NotImplementedError

    return lighting

def normalize_normal_inplace(normal, alpha):
    # normal: (3, H, W), alpha: (H, W)
    fg_mask = (alpha[None,...]>0.).repeat(3, 1, 1)
    normal = torch.where(fg_mask, torch.nn.functional.normalize(normal, p=2, dim=0), normal)

def render(viewpoint_camera, scene : Scene, 
           pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, 
           debug=False, rescale=1.0, attr_only=False,
           **kwargs
           ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if pipe.render_mode.find("defer") > -1:
        out_pbr = defer_render(viewpoint_camera, scene, pipe, bg_color, scaling_modifier,
                            debug, rescale, attr_only, pipe.hdr)
    else:
        raise NotImplementedError
    return out_pbr

def defer_render(viewpoint_camera, scene : Scene, pipe, bg_color : torch.Tensor, 
                 scaling_modifier = 1.0, debug=False, rescale=1.0, attr_only=False, hdr=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    test_flag = False
    pc = scene.gaussians
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=float(viewpoint_camera.image_width / 2),
        cy=float(viewpoint_camera.image_height / 2),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=True,
        debug=False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)


    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(opacity.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True) # (N, 3)
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    diffuse = pc.get_diffuse # (N, 3) 
    normal, delta_normal = pc.get_normal(dir_pp_normalized=dir_pp_normalized, return_delta=False) # (N, 3) 
    delta_normal_norm = delta_normal.norm(dim=1, keepdim=True) if delta_normal is not None else None
    specular  = pc.get_specular # (N, 3) 
    roughness = pc.get_roughness   # (N, 1) 
    metallic = pc.get_metallic  #* 0
    albedo = pc.get_albedo 

    normal_normed = 0.5*normal + 0.5  # range (-1, 1) -> (0, 1)
    render_extras = {"normal": normal_normed}
    if debug: 
        normal_axis = pc.get_minimum_axis
        normal_axis, _ = flip_align_view(normal_axis, dir_pp_normalized)
        normal_axis_normed = 0.5*normal_axis + 0.5  # range (-1, 1) -> (0, 1)
        render_extras.update({"normal_axis": normal_axis_normed})

    if delta_normal_norm is not None:
        render_extras.update({"delta_normal_norm": delta_normal_norm.repeat(1, 3)})
    render_extras.update({
        "pos": pc.get_xyz,
        "diffuse": diffuse, 
        "specular": specular, 
        "roughness": roughness, 
        "metallic": metallic,
        "albedo": albedo,
        })
    
    features = torch.cat([f for f in render_extras.values()], dim=-1)
    out_extras = {}
    (_, _, rendered_image, rendered_opacity, rendered_depth,
     rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, distortion, radii) = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = torch.ones_like(means3D),
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp, 
        features=features)
    out_values = torch.split(rendered_feature, [i.shape[1] for i in render_extras.values()])
    out_extras = {
        k: v for k, v in zip(render_extras.keys(), out_values)
    }
    out_extras["alpha"] = rendered_opacity
    out_extras["depth"] = rendered_depth
    out_extras["rendered_surface_xyz"] = rendered_surface_xyz
    out_extras["distortion"] = distortion
    if test_flag:
        viewdirs = viewpoint_camera.get_rays()
        out_extras['pos'] = viewpoint_camera.camera_center + viewdirs * rendered_depth

    for k in["normal", "normal_axis"] if debug else ["normal"]:
        if k in out_extras.keys():
            out_extras[k] = (out_extras[k] - 0.5) * 2. # range (0, 1) -> (-1, 1)
    if pipe.depth_normal:
        defer_normal = F.normalize(rendered_pseudo_normal.permute(1, 2, 0), dim=-1).reshape(1, 1, -1, 3)
    else:
        defer_normal = F.normalize(out_extras['normal'].permute(1, 2, 0), dim=-1).reshape(1, 1, -1, 3)
    defer_incidents = defer_vis = None
   
    out_extras['albedo'] = out_extras['albedo'] * rescale
    reflvec = None; brdf_pkg = None
    if attr_only is False:
        if pipe.render_mode.find('split_sum') > -1:
            gt_alpha_mask = viewpoint_camera.gt_alpha_mask.cuda()
            gt_alpha_mask[gt_alpha_mask>=0.5] = 1
            gt_alpha_mask[gt_alpha_mask<0.5] = 0
            pos = out_extras['pos'].permute(1, 2, 0).reshape(1, 1, -1, 3)
            
            rendered_image, brdf_pkg = pc.brdf_mlp.shade_nero(pos, defer_normal, 
                                        out_extras['albedo'].permute(1, 2, 0).reshape(1, 1, -1, 3), 
                                        out_extras['metallic'].permute(1, 2, 0).reshape(1, 1, -1, 1), 
                                        out_extras['roughness'].permute(1, 2, 0)[:, :, 0].reshape(1, 1, -1, 1), 
                                    viewpoint_camera.camera_center[None, None, :].repeat(
                                        int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 1
                                        ).reshape(1, 1, -1, 3), hdr==False)
        else:
            rendered_image, brdf_pkg = pc.brdf_mlp.shade(out_extras['pos'].permute(1, 2, 0).reshape(1, 1, -1, 3), 
                                            defer_normal, 
                                            out_extras['diffuse'].permute(1, 2, 0).reshape(1, 1, -1, 3), 
                                            out_extras['specular'].permute(1, 2, 0).reshape(1, 1, -1, 3), 
                                            out_extras['roughness'].permute(1, 2, 0)[:, :, 0].reshape(1, 1, -1, 1), 
                                        viewpoint_camera.camera_center[None, None, :].repeat(
                                            int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 1
                                            ).reshape(1, 1, -1, 3))
        rendered_image = rendered_image.view(int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 3).permute(2, 0, 1)
        
        rendered_image = rendered_image * out_extras["alpha"] + (1-out_extras["alpha"]) * bg_color[:, None, None]
    else:
        rendered_image = torch.ones((3, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)))
    # Render normal from depth image, and alpha blend with the background. 
    out_extras["normal_ref"] = rendered_pseudo_normal
    if debug:
        out_extras["normal_ref_cam"] = rendered_world2cam(viewpoint_camera, out_extras["normal_ref"], out_extras['alpha'][0], bg_color)
        out_extras["normal_axis_cam"] = rendered_world2cam(viewpoint_camera, out_extras["normal_axis"], out_extras['alpha'][0], bg_color)
    
    normalize_normal_inplace(out_extras["normal"], out_extras["alpha"][0])
    if debug:
        out_extras["normal_cam"] = rendered_world2cam(viewpoint_camera, out_extras["normal"], out_extras['alpha'][0], bg_color)

    if brdf_pkg and brdf_pkg.get("diffuse_light") is not None:
        diffuse_light = brdf_pkg['diffuse_light'].squeeze() 
        out_extras['diffuse_light'] = diffuse_light

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    out = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii, 
    }
    out.update(out_extras)
    return out
