import os
from os.path import join, dirname, basename, exists, splitext
import cv2
import numpy as np
import glob
import open3d as o3d
import torch


def batch_pixel2camera(intrinsics, depths):
    """
    :param intrinsics: (b, 3, 3)
    :param depths: (b, H, W)
    :return: points_camera (b, H*W, 3)
    """
    cx, cy, fx, fy = intrinsics[:, 0, 2], intrinsics[:, 1, 2], intrinsics[:, 0, 0], intrinsics[:, 1, 1]     # (b, )

    b, H, W = depths.shape
    u_base = np.tile(np.arange(W), (H, 1))[None]      # (1, H, W)
    v_base = np.tile(np.arange(H)[:, np.newaxis], (1, W))[None]       # (1, H, W)
    X = (u_base - cx[:, None, None]) * depths / fx[:, None, None]       # (b, H, W)
    Y = (v_base - cy[:, None, None]) * depths / fy[:, None, None]       # (b, H, W)
    coord_camera = np.stack((X, Y, depths), axis=-1).astype(np.float32)      # (b, H, W, 3)
    points_camera = coord_camera.reshape((b, -1, 3))    # (b, H*W, 3)

    return points_camera


def batch_camera2world(points_camera, pose):
    """
    :param points_camera: (b, N, 3)
    :param pose: (b, 4, 4)
    :return: points_world (b, N, 3)
    """
    points_local_homo = np.concatenate((points_camera, np.ones(points_camera.shape[:2], dtype=np.float32)[..., None]), axis=-1)     # (b, N, 4)
    points_world_homo = np.matmul(pose, points_local_homo.transpose(0, 2, 1)).transpose(0, 2, 1)        # (b, 4, 4) * (b, 4, N) = (b, 4, N) -> (b, N, 4)
    points_world = np.divide(points_world_homo, np.clip(points_world_homo[..., [-1]], 1e-8, np.inf))[..., :-1]         # (b, N, 3)
    return points_world


def parallel_world2cam_pixel(points_world, intrinsic, pose):
    """
    project N points to M images
    :param points_world: (N, 3)
    :param intrinsic: (M, 3, 3)
    :param pose: (M, 4, 4)
    :return:
    """
    points_world_homo = np.concatenate((points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)), 1)        # (N, 4)
    points_cam_homo = np.matmul(np.linalg.inv(pose)[None], points_world_homo[:, None, :, None])        # (1, M, 4, 4) @ (N, 1, 4, 1) = (N, M, 4, 1)
    points_cam_homo = points_cam_homo[..., 0]        # (N, M, 4)
    points_cam = np.divide(points_cam_homo, points_cam_homo[..., [-1]])[..., :-1]  # (N, M, 3)

    points_pixel_homo = np.matmul(intrinsic, points_cam[..., None])              # (M, 3, 3) @ (N, M, 3, 1) = (N, M, 3, 1)
    points_pixel_homo = points_pixel_homo[..., 0]           # (N, M, 3)
    points_pixel = np.divide(points_pixel_homo, np.clip(points_pixel_homo[..., [-1]], 1e-8, np.inf))[..., :-1].round().astype(int)  # (u, v) coordinate, (N, M, 2)

    return points_cam, points_pixel


def filter_by_population(points, filter_population, save_path):
    if points is None:
        points = np.loadtxt(save_path)

    if not isinstance(filter_population, list) and not isinstance(filter_population, tuple):
        filter_population = [filter_population]

    labels = points[..., -1]
    labels_unique = np.unique(labels)

    for popu in filter_population:
        labels_filter = labels.copy()

        for l in labels_unique:
            idx = labels_filter == l
            if idx.sum() < popu:
                labels_filter[idx] = 0

        rearange_label = np.unique(labels_filter, return_inverse=True)[1]
        points_filter = np.concatenate((points[..., :-1], rearange_label[:, None]), -1)

        points_path = save_path.replace('.pts', f'_amount{popu:.0f}.pts')
        np.savetxt(points_path, points_filter)
        print(f'save to {points_path}')



def get_color(color_path):
    color = cv2.imread(color_path).astype(np.float32)
    return color


def get_depth(color_path):
    depth_path = color_path.replace('color', 'depth').replace('.jpg', '.png')
    oldbase = basename(depth_path).split('_')
    oldbase[1] = oldbase[1].replace('i', 'd')
    newbase = '_'.join(oldbase)
    depth_path = join(dirname(depth_path), newbase)

    depth = cv2.imread(depth_path, -1).astype(np.float32)
    if depth.ndim != 2:
        depth = depth[..., 0]
    # set invalid distance to zero
    depth[depth == 65535] = 0
    # depth in millimeters, transfer to meters
    depth /= 4000
    return depth


def get_intrinsic(color_path):
    intrinsic = np.loadtxt(color_path.replace('color', 'intrinsic').replace('.jpg', '.txt')).astype(np.float32)
    return intrinsic


def get_pose(color_path):
    pose = np.loadtxt(color_path.replace('color', 'pose').replace('.jpg', '.txt')).astype(np.float32)
    return pose


def get_mask(color_path, text, thres_gsam):
    gsam_dir = join(dirname(dirname(color_path)), 'results', text,  f'{text}_gsam{thres_gsam:.2f}')
    mask_path = join(gsam_dir, f'maskraw_{basename(color_path).replace(".jpg", ".png")}')
    mask_raw = cv2.imread(mask_path, -1).astype(np.float32)
    return mask_raw


def get_score(color_path, text, thres_gsam):
    gsam_dir = join(dirname(dirname(color_path)), 'results', text, f'{text}_gsam{thres_gsam:.2f}')
    score_path = join(gsam_dir, f'score_{basename(color_path).replace(".jpg", ".png")}')
    score = cv2.imread(score_path, -1).astype(np.float32) / 1000
    return score


def get_points_from_openscene_pth(base_dir, scene_id):

    def read_pth(points_list):
        p_lst, c_lst, v_lst = [], [], []
        for points_path in points_list:
            points, color, vertex_labels = torch.load(points_path)
            p_lst.append(points)
            c_lst.append(color)
            v_lst.append(vertex_labels)

        p_lst = np.vstack(p_lst)
        c_lst = np.vstack(c_lst)
        v_lst = np.concatenate(v_lst)

        return p_lst, c_lst, v_lst

    points_list = sorted(glob.glob(join(base_dir, 'matterport_3d', '**', f'{scene_id}*.pth'), recursive=True))
    points = read_pth(points_list)[0]
    np.savetxt(join(base_dir, 'matterport_2d', scene_id, 'points.pts'), points)

