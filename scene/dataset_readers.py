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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from tqdm import tqdm
from scene.NVDIFFREC.util import simple_fibonacci_sphere_sampling

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    normal_image: np.array
    alpha_mask: np.array
    albedo_image: np.array = None
    scene_name: str = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

class ORBSceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    novel_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        root = os.path.split(images_folder)[0]
        image_name = os.path.basename(image_path).split(".")[0]
        mask_path = os.path.join(root, 'masks', f'{image_name}.png')
        if os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            name, ext = os.path.splitext(image_path)
            image_path = name + ext.lower()
            image = Image.open(image_path)
        if os.path.exists(mask_path) == True:
            mask = Image.open(mask_path)
        else:
            mask = Image.fromarray(np.ones_like(np.array(image)[:, :, 0]) * 255.)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, 
                              normal_image=None, alpha_mask=mask)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        if "camera_angle_x" in contents.keys():
            fovx = contents["camera_angle_x"] 
            fovy = contents.get("camera_angle_y")
        elif "camera_angle_x" in contents['frames'][0].keys():
            fovx = contents['frames'][0]["camera_angle_x"]
            fovy = contents['frames'][0].get("camera_angle_y")
        else:
            fovx = None
            

        frames = contents["frames"]
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            scene_name = frame.get('scene_name') 
            if scene_name is not None:
                image_path = '/' + os.path.join(*(path.split('/')[:-1]), scene_name, frame["file_path"] + extension)
            else:
                image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            alpha_mask = norm_data[:, :, 3]
            alpha_mask = Image.fromarray(np.array(alpha_mask*255.0, dtype=np.byte), "L")
            alpha_mask_path = os.path.splitext(image_path)[0] + '_alpha.png'
            if os.path.exists(alpha_mask_path):
                alpha_mask = Image.open(alpha_mask_path)
            # arr = np.concatenate([arr, norm_data[:, :, 3:4]], axis=-1)
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")

            normal_cam_name = os.path.join(path, frame["file_path"] + "_normal" + extension)
            normal_cam_name2 = os.path.join(path, *(frame["file_path"].split('/')[:-1] + ['normal.png']))
            albedo_path = os.path.join(path, *(frame["file_path"].split('/')[:-1] + ['albedo.png']))
            if os.path.exists(normal_cam_name) or os.path.exists(normal_cam_name2):
                normal_image = Image.open(normal_cam_name) if os.path.exists(normal_cam_name) else Image.open(normal_cam_name2)
                normal_im_data = np.array(normal_image.convert("RGBA"))
                normal_bg_mask = (normal_im_data==128).sum(-1)==3
                normal_norm_data = normal_im_data / 255.0
                normal_arr = normal_norm_data[:,:,:3] * normal_norm_data[:, :, 3:4] + bg * (1 - normal_norm_data[:, :, 3:4])
                normal_arr[normal_bg_mask] = 0
                normal_image = Image.fromarray(np.array(normal_arr*255.0, dtype=np.byte), "RGB")
            else:
                normal_image = None
            if os.path.exists(albedo_path):
                albedo_image = Image.open(albedo_path)
            else:
                albedo_image = None

            if fovx == None:
                focal_length = contents["fl_x"]
                FovY = focal2fov(focal_length, image.size[1])
                FovX = focal2fov(focal_length, image.size[0])
            else:
                if fovy is None:
                    fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy 
                FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], 
                            normal_image=normal_image, alpha_mask=alpha_mask, albedo_image=albedo_image, scene_name=scene_name))
            
    return cam_infos

import imageio
def read_exr(path):
    return imageio.imread(path, format='EXR')

def readCamerasFromTransformsHDR(path, transformsfile, white_background, extension=".exr"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        if "camera_angle_x" in contents.keys():
            fovx = contents["camera_angle_x"] 
            fovy = contents.get("camera_angle_y")
        elif "camera_angle_x" in contents['frames'][0].keys():
            fovx = contents['frames'][0]["camera_angle_x"]
            fovy = contents['frames'][0].get("camera_angle_y")
        else:
            fovx = None
            

        frames = contents["frames"]
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            scene_name = frame.get('scene_name') 
            if scene_name is not None:
                image_path = '/' + os.path.join(*(path.split('/')[:-1]), scene_name, frame["file_path"] + extension)
            else:
                image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            hdr = True if image_path.endswith('.exr') else False
            if hdr:
                image = read_exr(image_path)
                im_data = np.array(image)
                height, width = image.shape[:2]
            else:
                image = Image.open(image_path)
                im_data = np.array(image.convert("RGBA"))
                height, width = image.size
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data
            if not hdr:
                norm_data = im_data / 255.
            alpha_mask = im_data[:, :, 3] / 255.0
            arr = norm_data[:,:,:3] * alpha_mask[:, :, None] + bg * (1 - alpha_mask[:, :, None])
            image = arr.astype(np.float32)
            if not hdr:
                image = Image.fromarray(np.array(image*255.0, dtype=np.byte), "RGB")
            alpha_mask = Image.fromarray(np.array(alpha_mask*255.0, dtype=np.byte), "L")

            if fovx == None:
                focal_length = contents["fl_x"]
                FovY = focal2fov(focal_length, height)
                FovX = focal2fov(focal_length, width)
            else:
                if fovy is None:
                    fovy = focal2fov(fov2focal(fovx, width), height)
                FovY = fovy 
                FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=width, height=height, 
                            normal_image=None, alpha_mask=alpha_mask, albedo_image=None, scene_name=scene_name))
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", sphere_init=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        # test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")

    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readStanfordORBInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransformsHDR(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransformsHDR(path, "transforms_test.json", white_background, extension)
    print("Reading Novel Transforms")
    novel_cam_infos = readCamerasFromTransformsHDR(path, "transforms_novel.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        # test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")

    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = ORBSceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           novel_cameras=novel_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "StanfordORB" : readStanfordORBInfo,
}