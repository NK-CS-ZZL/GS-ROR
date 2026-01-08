import os
import numpy as np
import math
import json
import glob
import argparse
import pickle
import shutil
from skimage.io import imread, imsave
from utils.graphics_utils import normal_from_depth_image
import torch


'''
    Convert NeRO Blender dataset into NeRF format.
'''

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="data/glossy_relight/relight_gt", help="path to the GlossyBlender dataset")

    opt = parser.parse_args()
    scenes = os.listdir(opt.path)
    for scene in scenes:
        root = os.path.join(opt.path, scene)
        output_path = os.path.join(opt.path + "_blender", scene)
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        print(f'[INFO] Data from {root}')
        img_num = len(glob.glob(f'{root}/*.pkl'))
        img_ids = [str(k) for k in range(img_num)]
        cams = [read_pickle(f'{root}/{k}-camera.pkl') for k in range(img_num)]  # pose(3,4)  K(3,3)
        img_files = [f'{root}/{k}.png' for k in range(img_num)]
        depth_files = [f'{root}/{k}-depth.png' for k in range(img_num)]

        frames = []
        os.makedirs(f'{output_path}/depth', exist_ok=True)
        for image, cam, depth in zip(img_files, cams, depth_files):
            img_id = int(os.path.basename(image)[:-4])
            w2c = np.array(cam[0].tolist()+[[0,0,0,1]])
            c2w = np.linalg.inv(w2c)
            c2w[:3, 1:3] *= -1  # opencv -> blender/opengl
            frames.append({
                'file_path': os.path.join("rgb", os.path.basename(image)).replace(".png",""),
                'transform_matrix': c2w.tolist(),
            })
            depth = imread(depth)[:, :]
            depth = depth.astype(np.float32) / 65535 * 15
            mask = depth < 14.5
            normal = normal_from_depth_image(torch.from_numpy(depth).float(), 
                                             torch.from_numpy(cams[0][1]).float(), 
                                             torch.from_numpy(c2w).float())
            normal = torch.nn.functional.normalize(normal, dim = -1).numpy() * mask[:, :, None]
            normal = (normal + 1) / 2
            normal = (normal * 255).astype('uint8')
            imsave(f'{output_path}/depth/{img_id}_normal.png', normal)

        mask = (mask[...,None] * 255).astype(np.uint8)
        fl_x = float(cams[0][1][0,0])
        fl_y = float(cams[0][1][1,1])

        transforms = {
            'w': 800,
            'h': 800,
            'fl_x': fl_x,
            'fl_y': fl_y,
            'cx': 400,
            'cy': 400,
            # 'aabb_scale': 2,
            'frames': frames,
        }

        # write json
        json_out_path = os.path.join(output_path, f'transforms_train.json')
        print(f'[INFO] write to {json_out_path}')
        with open(json_out_path, 'w') as f:
            json.dump(transforms, f, indent=2)
        json_out_path = os.path.join(output_path, f'transforms_test.json')
        with open(json_out_path, 'w') as f:
            json.dump(transforms, f, indent=2)
        
        # write imgs
        img_out_path = os.path.join(output_path, "rgb")
        if not os.path.exists(img_out_path):
            os.makedirs(img_out_path, exist_ok=True)
        print(f'[INFO] Process rgbs')
        print(f'[INFO] write to {img_out_path}')
        for img_id in img_ids:
            depth = imread(f'{root}/{img_id}-depth.png')[:, :]
            depth = depth.astype(np.float32) / 65535 * 15
            mask = depth < 14.5
            mask = (mask[...,None] * 255).astype(np.uint8)

            image = imread(f'{root}/{img_id}.png')[..., :3]
            image = np.concatenate([image, mask], axis=-1)

            imsave(f'{img_out_path}/{img_id}.png', image)

        print(f"[INFO] Scece [{scene}] Finished.")




    
