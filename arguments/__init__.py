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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.random_background = True
        self.data_device = "cuda"
        self.eval = False
        self.env_mode = "envmap"
        self.envmap_res = 64
        self.use_delta = 1
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.render_mode = "defer+split_sum"
        self.render_radiance = False
        self.defer_radiance = False
        self.depth_normal = False
        self.gaussian_type = '3d'
        self.envmap_path = ''
        self.hdr = False

        
        super().__init__(parser, "Pipeline Parameters")
        self.brdf = False

    def extract(self, args):
        g = super().extract(args)
        self.render_mode = args.render_mode
        g.convert_SHs_python = True
        g.env_mode = args.env_mode
        return g

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.feature_lr_final = 0.000025
        self.opacity_lr = 0.05
        self.sdf_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.spe_mlp_lr = 0.001
        self.indirect_lr = 0.05
        
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.opacity_cull = 0.005
        self.brdf_mlp_lr_init = 1.6e-2
        self.brdf_mlp_lr_final = 1.6e-3 
        self.brdf_mlp_lr_delay_mult = 0.01
        self.brdf_mlp_lr_max_steps = 30_000
        self.normal_lr = 0.0002
        self.specular_lr = 0.0002
        self.albedo_lr = 0.0002
        self.diffuse_lr = 0.0002
        self.metallic_lr = 0.0002
        self.roughness_lr = 0.0002
        self.normal_reg_from_iter = 0
        self.normal_reg_util_iter = 30_000
        self.lambda_zero_one = 1e-3
        self.lambda_predicted_normal = 2e-1
        self.lambda_delta_reg = 1e-3
        self.lambda_opacity_reg = 0.
        self.fix_brdf_lr = 0
        
        self.opacity_01_from_iter = 0
        self.opacity_01_under_iter = 30_000
        self.opacity_remove_from_iter = 30_000
        self.opacity_remove_threshold = 0.99
        self.lambda_brdf_smoothness = 0.0
        self.lambda_base_smoothness = 0.0
        self.lambda_depth_brdf_smoothness = 0.0
        self.enable_vis_from_iteration = 30_000
        self.pbr_from_iter = 10_000
        self.zero_one_use_gt = False
        self.lambda_light_reg = 0.0
        self.sdf_remove_from_iter = 40000
        self.metallic = 1
        
        super().__init__(parser, "Optimization Parameters")

class TensoSDFOptimParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.lr_xyz_init = 1e-2
        self.lr_net_init = 1e-3
        self.lr_factor = 1.
        self.pre_lr_factor = 1.
        self.lambda_eikonal = 0.1
        self.lambda_hessian = 0.0005
        self.lambda_sparse = 0.05
        self.lambda_tv_sdf = 0.1
        self.lambda_occ = 0.
        self.lambda_rgb = 1.
        self.lambda_mask = 1
        self.lambda_sdf_large = 1.
        self.lambda_sdf_small = 1.
        self.lambda_depth_g2sdf = 0.5
        self.lambda_depth_sdf2g = 0.5
        self.lambda_normal_g2sdf = 0.5
        self.lambda_normal_sdf2g = 0.5
        self.lambda_gaussian = 1e-5
        self.lambda_radiance = 1.

        self.pretrain_sdf = ""

        self.sdf_from_iter = 1000
        self.batchify = False
        self.batch_size = 2048
        self.sdf_init_iters = 1000

        self.update_alpha_mask_list = [2000, 10000] # [2000, 10000]
        self.upsample_list = [2000, 10000] # [2000, 10000]

        self.sdf_mode = 'TensoSDF'
        self.decay = False
        self.lr_decay_target_ratio = 5e-2
        self.lr_decay_iters = 6000

        self.gridSize = 128
        
        super().__init__(parser, "TensoSDF Optimization Parameters", sentinel)


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
