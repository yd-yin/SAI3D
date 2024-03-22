import os
from os.path import join
import numpy as np
import glob
import tqdm
import argparse
import torch


def convert_scannet_to_openmask3d_format(data_dir, out_dir):
    '''
    Organize ScanNet dataset into the format that openmask3d can use
    Format for OpenMask3D:                                              
    ├── scene_0011_00
    │     ├── data 
    │     │      ├── intrinsic                 <- folder with the intrinsics
    │     │      └── pose                      <- folder with the poses
    │     ├── data_compressed                 
    │     │      ├── color                     <- folder with the color images
    │     │      └── depth                     <- folder with the depth images
    │     └── scene_0011_00_vh_clean_2.ply     <- path to the point cloud/mesh ply file
    '''
    data_dir = os.path.abspath(data_dir)
    out_dir = os.path.abspath(out_dir)
    with open(join(data_dir, 'Tasks/Benchmark/scannetv2_val.txt'), 'r') as f:
        val_scenes = f.readlines()
    val_scenes = [x.strip() for x in val_scenes]

    os.makedirs(out_dir, exist_ok=True)

    posed_img_dir = join(data_dir, 'posed_images')

    for scene_id in tqdm.tqdm(val_scenes):
        # print(scene_id)
        out_scene_dir = os.path.join(out_dir, scene_id)
        os.makedirs(out_scene_dir, exist_ok=True)

        old_data_dir = os.path.join(posed_img_dir, scene_id)

        new_data_dir = os.path.join(out_dir, scene_id, 'data')
        new_data_compressed_dir = os.path.join(out_dir, scene_id, 'data_compressed')

        os.makedirs(new_data_dir, exist_ok=True)
        os.makedirs(new_data_compressed_dir, exist_ok=True)

        # 建立软链接
        out_intrinsic_dir = join(new_data_dir, 'intrinsic')
        os.makedirs(out_intrinsic_dir, exist_ok=True)
        intrinsic_path = join(old_data_dir, 'intrinsic_color.txt')
        print(intrinsic_path)
        os.symlink(intrinsic_path, join(out_intrinsic_dir, 'intrinsic_color.txt'))

        out_poses_dir = join(new_data_dir, 'pose')
        out_color_dir = join(new_data_compressed_dir, 'color')
        out_depth_dir = join(new_data_compressed_dir, 'depth')
        os.makedirs(out_poses_dir, exist_ok=True)
        os.makedirs(out_color_dir, exist_ok=True)
        os.makedirs(out_depth_dir, exist_ok=True)

        color_paths = sorted(glob.glob(join(old_data_dir, '*.jpg')))
        for i, color_path in enumerate(color_paths):
            os.symlink(color_path, join(out_color_dir, f'{i}.jpg'))
            depth_path = color_path.replace('.jpg', '.png')
            os.symlink(depth_path, join(out_depth_dir, f'{i}.png'))
            pose_path = color_path.replace('.jpg', '.txt')
            os.symlink(pose_path, join(out_poses_dir, f'{i}.txt'))

        os.symlink(join(data_dir, 'scans', scene_id, f'{scene_id}_vh_clean_2.ply'),
                   join(os.path.dirname(new_data_dir), f'{scene_id}_vh_clean_2.ply'))


def convert_masks_to_openmask3d_format(mask_dir, out_dir):
    '''
    Convert the predicted class-agnostic masks from ScanNet benchmark format to openmask3d format
    ScanNet benchmark format: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py
    OpenMask3D format: .pt file with shape (num_points, num_masks)
    '''
    def read_masks(masks_path):
        with open(masks_path, 'r') as f:
            lines = f.readlines()
        num_masks = len(lines)
        masks = []
        for line in tqdm.tqdm(lines, desc='Loading masks'):
            rel_ins_path, label_id, confidence = line.split(' ')
            ins_path = os.path.join(os.path.dirname(masks_path), rel_ins_path)
            mask = np.loadtxt(ins_path)
            masks.append(mask)
        masks = np.array(masks).transpose(1, 0)
        assert num_masks == masks.shape[1]
        return torch.tensor(masks)

    os.makedirs(out_dir, exist_ok=True)
    masks_paths = glob.glob(join(mask_dir, 'scene*.txt'))
    for masks_path in tqdm.tqdm(masks_paths):
        print(masks_path)
        masks = read_masks(masks_path)
        out_path = join(out_dir, os.path.basename(masks_path).replace('.txt', '_masks.pt'))
        torch.save(masks, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--app', type=int, required=True,
                        help='0: convert scannet to openmask3d format, 1: convert masks to openmask3d format')
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    if args.app == 0:
        convert_scannet_to_openmask3d_format(args.base_dir, args.out_dir)
    elif args.app == 1:
        convert_masks_to_openmask3d_format(args.base_dir, args.out_dir)
    else:
        raise ValueError('Invalid app')
