import os
from os.path import join, dirname, basename, exists, splitext
import cv2
import numpy as np
import glob
import open3d as o3d
import torch
import json
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
import plyfile

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
================================================================================================
general utils
================================================================================================
"""
def get_points_from_ply(ply_path):
    ply = plyfile.PlyData.read(ply_path)
    xs = np.array(ply["vertex"].data['x'])[:, None]
    ys = np.array(ply["vertex"].data['y'])[:, None]
    zs = np.array(ply["vertex"].data['z'])[:, None]
    points = np.concatenate((xs, ys, zs), axis=-1)
    np.savetxt(join(dirname(ply_path), 'points.pts'), points)

def get_splits(base_dir, scannetpp):
    if scannetpp:
        train_split_path = join(base_dir, 'splits/nvs_sem_train.txt')
        val_split_path = join(base_dir, 'splits/nvs_sem_val.txt')
    else:
        train_split_path = join(
            base_dir, 'Tasks/Benchmark/scannetv2_train.txt')
        val_split_path = join(base_dir, 'Tasks/Benchmark/scannetv2_val.txt')

    with open(train_split_path, 'r') as f:
        train_split = f.readlines()
    train_split = [s.strip() for s in train_split]

    with open(val_split_path, 'r') as f:
        val_split = f.readlines()
    val_split = [s.strip() for s in val_split]

    return train_split, val_split

def construct_saving_name(args):
    save_name = \
        f'points_objness_label_{args.mask_name}_connect({args.thres_connect[0]},{args.thres_connect[-1]},{len(args.thres_connect)}).pts'
    if (args.thres_trunc > 0):
        save_name = save_name.replace('.pts', f'trunc{args.thres_trunc}.pts')
    save_name = args.similar_metric + '_' + save_name
    if (args.max_neighbor_distance is not None):
        save_name = save_name.replace('.pts', f'_depth{args.max_neighbor_distance}.pts')
    if (args.thres_merge > 0):
        save_name = f'merge{args.thres_merge}_' + save_name
    if (args.text is not None):
        save_name = f'{args.text}_' + save_name
    if (args.test):
        save_name = 'test_' + save_name

    return save_name


def num_to_natural(group_ids, void_number=-1):
    """
    code credit: SAM3D
    """
    if (void_number == -1):
        # [-1,-1,0,3,4,0,6] -> [-1,-1,0,1,2,0,3]
        if np.all(group_ids == -1):
            return group_ids
        array = group_ids.copy()

        unique_values = np.unique(array[array != -1])
        mapping = np.full(np.max(unique_values) + 2, -1)
        # map ith(start from 0) group_id to i
        mapping[unique_values + 1] = np.arange(len(unique_values))
        array = mapping[array + 1]

    elif (void_number == 0):
        # [0,3,4,0,6] -> [0,1,2,0,3]
        if np.all(group_ids == 0):
            return group_ids
        array = group_ids.copy()

        unique_values = np.unique(array[array != 0])
        mapping = np.full(np.max(unique_values) + 2, 0)
        mapping[unique_values] = np.arange(len(unique_values)) + 1
        array = mapping[array]
    else:
        raise Exception("void_number must be -1 or 0")

    return array


def draw_adj_distribuution(adj, save_path):
    left = [0.02*i for i in range(50)]

    similar, count = np.unique(adj, return_counts=True)
    similar = similar[1:]  # exclude 0
    count = count[1:]
    bra_count = np.zeros(50)
    bra_loc = similar//0.02

    for i in range(50):
        bra_count[i] = count[bra_loc == i].sum()

    plt.bar(left, bra_count, width=0.02)
    plt.savefig(save_path)


def save_seg_to_point_obj_label(args):
    scene_seg_path = join(args.base_dir, 'scans', args.scene_id,
                          f'{args.scene_id}_vh_clean_2.0.010000.segs.json')
    with open(scene_seg_path, 'r') as f:
        seg_data = json.load(f)
    instance_ids = np.array(seg_data['segIndices'])[:, None]

    points_path = join(args.base_dir, 'scans', args.scene_id, 'points.pts')
    with open(points_path, 'r') as f:
        points = np.loadtxt(f)

    objness = np.ones_like(instance_ids)  # set obj to 1

    points_obj_label = np.concatenate((points, objness, instance_ids), axis=-1)
    save_path = join(args.base_dir, 'scans', args.scene_id, 'results', f'seg_points_objness_label.pts')
    print(f'saving to {save_path}')
    np.savetxt(save_path, points_obj_label)


"""
================================================================================================
utils for affinity calculation
================================================================================================
"""


def get_similar_confidence_matrix_handle(seg_neighbors,
                                         seg_ids,
                                         seg_seen,
                                         points_label,
                                         similar_metric,
                                         thres_trunc):
    view_num = seg_seen.shape[1]
    seg_num = seg_seen.shape[0]

    similar_sum = np.zeros([seg_num, seg_num], dtype=np.float32)
    confidence_sum = np.zeros([seg_num, seg_num], dtype=np.float32)

    one_view_similar = np.zeros([seg_num, seg_num], dtype=np.float32)
    one_view_confidence = np.zeros([seg_num, seg_num], dtype=np.float32)

    for m in tqdm(range(view_num)):
        plabels = points_label[:, m]  # (N,)
        one_view_similar.fill(0.)
        one_view_confidence.fill(0.)

        label_range = int(np.max(plabels) + 1)
        if (label_range < 2):
            continue  # there's no valid mask in the view

        # get label distribution of all segs in the view
        seglabels = np.zeros([seg_num, label_range], dtype=np.float32)  # (s,lr)

        p_labels_segids = np.stack([plabels, seg_ids], axis=1)
        unique_labels_segids, unique_counts = np.unique(p_labels_segids, return_counts=True, axis=0)

        unique_labels_segids = unique_labels_segids.astype(np.int32)
        seglabels[unique_labels_segids[:, 1], unique_labels_segids[:, 0]] = unique_counts  # (s,lr)

        # exclude invalid label
        nonzero_seglabels = seglabels[:, 1:].copy()  # (s,lr-1)

        # normalize the label feature, so that nonzero_seglabels[i,j], demonsttrates the percentage of members with label=j+1 in primitive i
        if (similar_metric == "2-norm"):
            nonzero_seglabels = np.divide(
                nonzero_seglabels, 
                np.clip(np.linalg.norm(nonzero_seglabels, axis=-1), 1e-8, np.inf)[:, None])
        elif (similar_metric == "1-norm" or similar_metric == "Hellinger"):
            nonzero_seglabels = np.divide(
                nonzero_seglabels, 
                np.clip(np.linalg.norm(nonzero_seglabels, ord=1, axis=-1), 1e-8, np.inf)[:, None])
        elif (similar_metric == "inf-norm"):
            max_id = np.argmax(nonzero_seglabels, axis=-1)  # (s,)
            nonzero_seglabels = np.zeros_like(nonzero_seglabels)  # (s,lr-1)
            nonzero_seglabels[np.arange(seg_num), max_id] = 1  # (s,lr-1)

        # in every iter, we process one seg and its all neighbors
        for i in range(seg_neighbors.shape[0]):
            if (seg_neighbors[i].nonzero()[0].size == 0):
                continue
            neighbors_labels = nonzero_seglabels[seg_neighbors[i]]
            one_view_similar[i, seg_neighbors[i]] = calcu_similar(nonzero_seglabels[i],
                                                                  neighbors_labels,
                                                                  similar_metric=similar_metric,
                                                                  thres_trunc=thres_trunc)
            # (nei,) * () = (nei,)
            one_view_confidence[i, seg_neighbors[i]] = seg_seen[seg_neighbors[i], m]*seg_seen[i, m]

        similar_sum = similar_sum + one_view_similar*one_view_confidence
        confidence_sum += one_view_confidence

    return [similar_sum, confidence_sum]


def multiprocess_get_similar_confidence_matrix(seg_seen, 
                                               seg_neighbors, 
                                               seg_ids, 
                                               points_label, 
                                               similar_metric, 
                                               thres_trunc=0., 
                                               process_num=8):
    """
    :param seg_seen: (S, M), ratio of seen part of primitives in every view
    :param seg_neighbors: (S, S), 1 if two segs are neighbors
    :param seg_ids: (N, ), seg id of every point
    :param points_label: (N, M), labels of all points in all views

    :return similar: wighted sum of how much the two primitives are similar in every view
    :return confidence: sum of wight of how much we can trust the similar score in every view
    """
    seg_num = seg_seen.shape[0]
    M = seg_seen.shape[1]

    similar_sum = np.zeros([seg_num, seg_num], dtype=np.float32)
    confidence_sum = np.zeros([seg_num, seg_num], dtype=np.float32)

    view_per_process = (M + process_num - 1) // process_num

    pool = multiprocessing.Pool(process_num)
    res_list = []
    for i in range(process_num):
        process_seg_seen = seg_seen[:,i*view_per_process:(i+1)*view_per_process]  # (s,b)
        # (N,b)
        process_plabels = points_label[:,i*view_per_process:(i+1)*view_per_process]
        res = pool.apply_async(get_similar_confidence_matrix_handle,
                               args=(seg_neighbors, seg_ids,
                                     process_seg_seen, process_plabels,
                                     similar_metric,
                                     thres_trunc))
        res_list.append(res)

    for i in tqdm(range(process_num)):
        process_similar_sum, process_confidence_sum = res_list[i].get()
        similar_sum += process_similar_sum
        confidence_sum += process_confidence_sum

    return [similar_sum, confidence_sum]


@torch.inference_mode()
def torch_get_similar_confidence_matrix(seg_neighbors,
                                        seg_ids,
                                        seg_seen,
                                        points_label,
                                        similar_metric,
                                        thres_trunc=0):
    """
    :param seg_seen: (S, M), ratio of seen part of primitives in every view
    :param seg_neighbors: (S, S), 1 if two segs are neighbors
    :param seg_ids: (N, ), seg id of every point
    :param points_label: (N, M), labels of all points in all views

    :return similar: wighted sum of how much the two primitives are similar in every view
    :return confidence: sum of wight of how much we can trust the similar score in every view
    """
    view_num = seg_seen.shape[1]
    seg_num = seg_seen.shape[0]

    print("preparing data on gpu")
    gpu_seg_ids = torch.tensor(seg_ids, device=device, dtype=torch.int32)  # (n,)
    gpu_points_label = torch.tensor(points_label, device='cpu', dtype=torch.float32)  # (n,m)
    gpu_seg_neighbors = torch.tensor(seg_neighbors, device=device, dtype=torch.bool)  # (s,s)
    gpu_seg_seen = torch.tensor(seg_seen, device=device, dtype=torch.float32)  # (s,m)

    similar_sum = torch.zeros([seg_num, seg_num], device=device, dtype=torch.float32)  # (s,s)
    confidence_sum = torch.zeros([seg_num, seg_num], device=device, dtype=torch.float32)
    one_view_similar = torch.zeros([seg_num, seg_num], device=device, dtype=torch.float32)
    one_view_confidence = torch.zeros([seg_num, seg_num], device=device, dtype=torch.float32)

    for m in tqdm(range(view_num)):

        plabels = gpu_points_label[:, m].to(device)  # (N,)
        one_view_similar.fill_(0.)
        one_view_confidence.fill_(0.)

        label_range = int(torch.max(plabels) + 1)
        if (label_range < 2):
            continue  # there's no valid mask in the view

        # get label distribution of all segs in the view
        seglabels = torch.zeros([seg_num, label_range], device=device, dtype=torch.float32)  # (s,lr)

        p_labels_segids = torch.stack([plabels, gpu_seg_ids], dim=1)
        unique_labels_segids, unique_counts = \
            torch.unique(p_labels_segids, return_counts=True, dim=0)

        unique_labels_segids = unique_labels_segids.type(torch.long)
        unique_counts = unique_counts.type(torch.float32)
        seglabels[unique_labels_segids[:, 1], unique_labels_segids[:, 0]] = unique_counts  # (s,lr)

        # exclude invalid label
        nonzero_seglabels = seglabels[:, 1:]  # (s,lr-1)

        # normalize the label feature, so that nonzero_seglabels[i,j], demonsttrates the percentage of members with label=j+1 in primitive i
        if (similar_metric == "2-norm"):
            nonzero_seglabels = torch.divide(
                nonzero_seglabels, 
                torch.clamp(torch.norm(nonzero_seglabels, dim=-1), 1e-8)[:, None])
        else:
            raise NotImplementedError
        del unique_counts, unique_labels_segids
        del p_labels_segids, plabels

        # in every iter, we process one batch primitives aand its all neighbors
        batch_size = 200
        for start_id in range(0, seg_num, batch_size):
            if (seg_neighbors[start_id:start_id+batch_size].nonzero()[0].size == 0):
                continue
            all_neighbors_mask = \
                torch.sum(gpu_seg_neighbors[start_id:start_id+batch_size], dim=0) > 0
            
            neighbors_labels = nonzero_seglabels[all_neighbors_mask]
            
            one_view_similar[start_id:start_id+batch_size, all_neighbors_mask] = \
                torch_calcu_all_similar(nonzero_seglabels[start_id:start_id+batch_size],
                                        neighbors_labels,
                                        similar_metric=similar_metric,
                                        thres_trunc=thres_trunc)
        # (s,1) @ (1,s) = (s,s)
        one_view_confidence = gpu_seg_seen[:,m][:, None] @ gpu_seg_seen[:, m][None, :]

        del seglabels, nonzero_seglabels

        confidence_sum += one_view_confidence
        similar_sum += (one_view_similar*one_view_confidence)

    return [similar_sum.cpu().numpy(), confidence_sum.cpu().numpy()]


def calcu_similar(label0, labels1, similar_metric, thres_trunc=0):
    '''
    :param label0: (lr-1,), label distribuion of one seg 
    :param labels1: (neigh,lr-1), label distribution of neighbors of the seg above
    '''
    # choose a function, f(0) = 1 , f(2) = 0 , f' < 0 ,  f'' < 0
    if (similar_metric == '2-norm'):
        # (neigh) from 0. to sqrt(2.)
        dis = np.linalg.norm((labels1 - label0), axis=-1)
        def similar_func(x): return (2-x**2)/2.
        similar = similar_func(dis)
    elif (similar_metric == '1-norm'):
        # (neigh) from 0. to 2.
        dis = np.linalg.norm((labels1 - label0), ord=1, axis=-1)
        def similar_func(x): return (2. - x)/2.
        similar = similar_func(dis)
    elif (similar_metric == 'inf-norm'):
        # one-hot label vec
        dis = np.linalg.norm((labels1 - label0), ord=1, axis=-1)  # 0(equal) or 2(inequal)

        def similar_func(x): return (2. - x)/2.  # f(0) = 1, f(2) = 0
        similar = similar_func(dis)
    elif (similar_metric == 'Hellinger'):
        hellinger_div = np.linalg.norm((labels1**0.5 - label0**0.5), axis=-1) / (2**0.5)
        similar = 1.-hellinger_div
    else:
        assert 0, 'invalid similarity metric!'

    if (thres_trunc > 0):
        similar[similar < thres_trunc] = 0

    return similar


@torch.inference_mode()
def torch_calcu_all_similar(labels1, labels2, similar_metric="2-norm", thres_trunc=0):
    '''
    :param labels0: (s1, lr-1), label distribuion of one seg 
    :param labels1: (s2, lr-1), label distribution of neighbors of the seg above
    :return similar: (s1,s2) pairwise similar
    '''
    s1 = labels1.shape[0]
    s2 = labels2.shape[0]
    index1 = torch.tile(torch.arange(s1)[:, np.newaxis], (1, s2))  # (s1,s2)
    index2 = torch.tile(torch.arange(s2), (s1, 1))  # (s1,s2)
    if (similar_metric == '2-norm'):
        # (s1,s2,lr-1)->(s1,s2)
        dis = torch.norm(labels1[index1] - labels2[index2], dim=-1)
        def similar_func(x): return (2-x**2)/2.  # equal to cosine similarity
    else:
        raise NotImplementedError

    similar = similar_func(dis)
    if (thres_trunc > 0):
        similar[similar < thres_trunc] = 0

    return similar


"""
================================================================================================
utils for transformation
================================================================================================
"""


def world2cam_pixel(points_world, color_intrinsic, depth_intrinsic, pose):
    """project N points to M images

    :param points_world: (N, 3)
    :param color_intrinsic, depth_intrinsic: (M, 3, 3) the intrinsics of color and depth camera
    :param pose: (M, 4, 4)
    :return: points_cam(N,M,3), color_points_pixel(N,M,2), depth_points_pixel(N,M,2)
    """
    points_world_homo = \
        np.concatenate((points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)), 1)        # (N, 4)

    # (1, M, 4, 4) @ (N, 1, 4, 1) = (N, M, 4, 1)
    points_cam_homo = np.matmul(np.linalg.inv(pose)[None], points_world_homo[:, None, :, None])
    points_cam_homo = points_cam_homo[..., 0]        # (N, M, 4)

    points_cam = np.divide(
        points_cam_homo, points_cam_homo[..., [-1]])[..., :-1]  # (N, M, 3)

    # (M, 3, 3) @ (N, M, 3, 1) = (N, M, 3, 1)
    color_points_pixel_homo = np.matmul(color_intrinsic, points_cam[..., None])
    depth_points_pixel_homo = np.matmul(depth_intrinsic, points_cam[..., None])

    # (N, M, 3)
    color_points_pixel_homo = color_points_pixel_homo[..., 0]
    depth_points_pixel_homo = depth_points_pixel_homo[..., 0]

    color_points_pixel = \
        np.divide(color_points_pixel_homo, np.clip(color_points_pixel_homo[..., [-1]], 1e-8, np.inf))[..., :-1].round().astype(int)  # (u, v) coordinate, (N, M, 2)
    depth_points_pixel = \
        np.divide(depth_points_pixel_homo, np.clip(depth_points_pixel_homo[..., [-1]], 1e-8, np.inf))[..., :-1].round().astype(int)

    return points_cam, color_points_pixel, depth_points_pixel


@torch.inference_mode()
def torch_world2cam_pixel(points_world_all: np.array, color_intrinsic: np.array, depth_intrinsic: np.array, pose: np.array):
    """project N (1e6) points to M (1e3) images

    :param points_world: (N, 3)
    :param color_intrinsic, depth_intrinsic: (M, 3, 3) the intrinsics of color and depth camera
    :param pose: (M, 4, 4)
    :return: points_cam(n,M,3), color_points_pixel(n,M,2), depth_points_pixel(n,M,2)
    """
    batch_size = 10000

    color_intrinsic = torch.tensor(color_intrinsic, device=device, dtype=torch.float32)
    depth_intrinsic = torch.tensor(depth_intrinsic, device=device, dtype=torch.float32)
    pose = torch.tensor(pose, device=device, dtype=torch.float32)
    pose_inv = torch.linalg.inv(pose)   # (M, 4, 4)
    del pose

    N = points_world_all.shape[0]
    M = pose_inv.shape[0]
    # final_points_cam = torch.empty((N, M, 3), device='cpu', dtype=torch.float32)
    # final_color_points_pixel = torch.empty((N, M, 2), device='cpu', dtype=torch.int32)
    # final_depth_points_pixel = torch.empty((N, M, 2), device='cpu', dtype=torch.int32)
    final_points_cam = np.zeros((N, M, 3), dtype=np.float32)
    final_color_points_pixel = np.zeros((N, M, 2), dtype=int)
    final_depth_points_pixel = np.zeros((N, M, 2), dtype=int)

    for batch_start in tqdm(range(0, points_world_all.shape[0], batch_size)):
        # print("Migrating to CUDA")
        points_world = \
            torch.tensor(points_world_all[batch_start: batch_start+batch_size], device=device, dtype=torch.float32)
        points_world_homo = \
            torch.cat((points_world, torch.ones((points_world.shape[0], 1), dtype=torch.float32, device=device)), 1)

        points_cam_homo = torch.matmul(pose_inv[None], points_world_homo[:, None, :, None])
        points_cam_homo = points_cam_homo[..., 0]        # (N, M, 4)

        points_cam = torch.div(
            points_cam_homo[..., :-1], points_cam_homo[..., [-1]])  # (N, M, 3)

        # (M, 3, 3) @ (N, M, 3, 1) = (N, M, 3, 1)
        color_points_pixel_homo = torch.matmul(
            color_intrinsic, points_cam[..., None])
        depth_points_pixel_homo = torch.matmul(
            depth_intrinsic, points_cam[..., None])

        # (N, M, 3)
        color_points_pixel_homo = color_points_pixel_homo[..., 0]
        depth_points_pixel_homo = depth_points_pixel_homo[..., 0]

        color_points_pixel = \
            torch.div(color_points_pixel_homo[..., :-1], torch.clip(color_points_pixel_homo[..., [-1]], min=1e-8)).round().to(torch.int32)  # (u, v) coordinate, (N, M, 2)
        depth_points_pixel = \
            torch.div(depth_points_pixel_homo[..., :-1], torch.clip(depth_points_pixel_homo[..., [-1]], min=1e-8)).round().to(torch.int32)

        final_points_cam[batch_start: batch_start + batch_size] = points_cam.cpu().numpy()
        final_color_points_pixel[batch_start: batch_start + batch_size] = color_points_pixel.cpu().numpy()
        final_depth_points_pixel[batch_start: batch_start + batch_size] = depth_points_pixel.cpu().numpy()

    torch.cuda.empty_cache()
    # return (final_points_cam.cpu().numpy(), final_color_points_pixel.cpu().numpy(), final_depth_points_pixel.cpu().numpy())
    return (final_points_cam, final_color_points_pixel, final_depth_points_pixel)


def batch_pixel2camera(intrinsics, depths):
    """unproject pixels to camera coordinate

    :param intrinsics: (b, 3, 3)
    :param depths: (b, H, W)
    :return: points_camera (b, H*W, 3)
    """
    cx, cy, fx, fy = \
        intrinsics[:, 0, 2], intrinsics[:, 1, 2], intrinsics[:, 0, 0], intrinsics[:, 1, 1]     # (b, )

    b, H, W = depths.shape
    u_base = np.tile(np.arange(W), (H, 1))[None]      # (1, H, W)
    v_base = np.tile(np.arange(H)[:, np.newaxis], (1, W))[None]       # (1, H, W)
    X = (u_base - cx[:, None, None]) * depths / fx[:, None, None]       # (b, H, W)
    Y = (v_base - cy[:, None, None]) * depths / fy[:, None, None]       # (b, H, W)
    coord_camera = np.stack((X, Y, depths), axis=-1).astype(np.float32)      # (b, H, W, 3)
    points_camera = coord_camera.reshape((b, -1, 3))    # (b, H*W, 3)

    return points_camera


def batch_camera2world(points_camera, pose):
    """transform points from camera coordinate to world coordinate

    :param points_camera: (b, N, 3)
    :param pose: (b, 4, 4)
    :return: points_world (b, N, 3)
    """
    points_local_homo = np.concatenate((points_camera, np.ones(points_camera.shape[:2], dtype=np.float32)[..., None]), axis=-1)     # (b, N, 4)
    points_world_homo = np.matmul(pose, points_local_homo.transpose(0, 2, 1)).transpose(0, 2, 1)        # (b, 4, 4) * (b, 4, N) = (b, 4, N) -> (b, N, 4)
    points_world = np.divide(points_world_homo, np.clip(points_world_homo[..., [-1]], 1e-8, np.inf))[..., :-1]         # (b, N, 3)
    return points_world


"""
================================================================================================
utils for matterport dataset
================================================================================================
"""


def get_matterport_color(color_path):
    color = cv2.imread(color_path).astype(np.float32)
    return color


def get_matterport_depth(color_path):
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


def get_matterport_intrinsic(color_path):
    intrinsic = np.loadtxt(color_path.replace('color', 'intrinsic').replace('.jpg', '.txt')).astype(np.float32)
    return intrinsic


def get_matterport_pose(color_path):
    pose = np.loadtxt(color_path.replace('color', 'pose').replace('.jpg', '.txt')).astype(np.float32)
    return pose


def get_matterport_mask(base_dir, scene_id, color_name, mask_name, region_id='region4'):
    mask_dir = join(base_dir, '2D_masks', scene_id)
    mask_path = join(mask_dir, mask_name, f'maskraw_{color_name.replace(".jpg", ".png")}')
    mask_raw = cv2.imread(mask_path, -1).astype(np.float32)
    # mask_color_path = join(mask_dir, mask_name, f'maskcolor_{basename(color_path).replace(".jpg", ".png")}')
    # dest_dir = join(mask_dir, mask_name, region_id)
    # mask_color_dest_path = join(dest_dir, f'maskcolor_{basename(color_path).replace(".jpg", ".png")}')
    # color_dest_path = join(dest_dir, basename(color_path))

    # shutil.copy(mask_color_path,mask_color_dest_path)
    # shutil.copy(color_path,color_dest_path)

    return mask_raw


def get_matterport_semantic_mask(color_path, semantic):
    semantic_dir = join(dirname(dirname(color_path)), 'results', 'everything', 'semantic', semantic)
    semantic_path = join(semantic_dir, f'{basename(color_path).replace(".jpg", ".png")}')
    # from 0 to category_num, 0 means no class
    semantic_mask = cv2.imread(semantic_path, -1).astype(np.float32)
    # print(np.unique(semantic_mask))
    return semantic_mask


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

    points_list = sorted(glob.glob(
        join(base_dir, 'matterport_3d', '**', f'{scene_id}*.pth'), recursive=True))
    points = read_pth(points_list)[0]
    np.savetxt(join(base_dir, 'matterport_2d', scene_id, 'points.pts'), points)


"""
================================================================================================
utils for scannet dataset
================================================================================================
"""


def get_scannet_depth_mask(color_path, base_dir, scene_id, mask_group_name):
    mask_dir = join(base_dir, '2D_masks', scene_id, 'depth_' + mask_group_name)
    color_name = basename(color_path)
    # mask_name = 'maskraw_' + color_name.replace('jpg','png')
    mask_name = 'maskraw_' + color_name.replace('jpg', 'png')
    mask_path = join(mask_dir, mask_name)
    mask_raw = cv2.imread(mask_path, -1).astype(np.float32)
    # print("mask:",mask_raw.shape)  #(968, 1296)
    return mask_raw


def get_scannet_pose(color_path):
    pose = np.loadtxt(color_path.replace('.jpg', '.txt')).astype(np.float32)
    return pose


def get_scannet_depth(color_path):
    depth_path = color_path.replace('.jpg', '.png')
    depth = cv2.imread(depth_path, -1).astype(np.float32)
    # print(depth[depth>0])
    depth /= 1000.
    # print(depth.shape) #(480,640)
    return depth


def get_scannet_color_and_depth_intrinsic(color_path):
    color_intrinsic_path = join(dirname(color_path), 'intrinsic_color.txt')
    depth_intrinsic_path = join(dirname(color_path), 'intrinsic_depth.txt')

    color_intrinsic = np.loadtxt(color_intrinsic_path).astype(np.float32)[:3, :3]
    depth_intrinsic = np.loadtxt(depth_intrinsic_path).astype(np.float32)[:3, :3]

    return color_intrinsic, depth_intrinsic


def get_scannet_mask(color_path, base_dir, scene_id, mask_group_name):
    mask_dir = join(base_dir, '2D_masks', scene_id, mask_group_name)
    color_name = basename(color_path)
    mask_name = 'maskraw_' + color_name.replace('jpg', 'png')
    mask_path = join(mask_dir, mask_name)
    mask_raw = cv2.imread(mask_path, -1).astype(np.float32)
    # print("mask:",mask_raw.shape)  #(968, 1296)
    return mask_raw


def get_scannet_semantic_mask(color_path, base_dir, scene_id, is_scannet200):
    if (not is_scannet200):
        mask_dir = join(base_dir, '2D_masks', scene_id, 'ovseg')
    else:
        mask_dir = join(base_dir, '2D_masks', scene_id, 'ovseg200')
    color_name = basename(color_path)
    mask_name = 'semantic_maskraw_' + color_name.replace('jpg', 'png')
    mask_path = join(mask_dir, mask_name)
    # print(mask_path)
    mask_raw = cv2.imread(mask_path, -1).astype(np.float32)
    # print("mask:",mask_raw.shape)  #(968, 1296)
    return mask_raw


"""
================================================================================================
utils for scannetpp dataset
================================================================================================
"""


def get_scannetpp_poses_and_intrinsics(base_dir, scene_id, freq):
    pose_dir = join(base_dir, 'posed_images', scene_id)
    pose_path = join(pose_dir, 'pose_intrinsic_imu.json')
    with open(pose_path, 'r') as f:
        data = json.load(f)
    frame_num = len(data.keys())
    poses = []
    intrinsics = []
    for frame in range(0, frame_num, freq):
        key = 'frame_%06d' % frame
        pose = np.array(data[key]["aligned_pose"]).reshape(4, 4).astype(np.float32)
        intrinsic = np.array(data[key]['intrinsic']).astype(np.float32)
        poses.append(pose)
        intrinsics.append(intrinsic)

    return poses, intrinsics


def get_scannetpp_depth(color_path, color_shape):
    depth_dir = join(dirname(dirname(color_path)), 'depth')
    depth_path = join(depth_dir, basename(color_path).replace('jpg', 'png'))

    depth = cv2.imread(depth_path, -1).astype(np.float32)
    depth = cv2.resize(depth, [color_shape[1], color_shape[0]], interpolation=1)
    # print(depth[depth>0])
    depth /= 1000.
    # print('depth:', depth.shape) #(480,640)
    return depth


def get_scannetpp_mask(color_path, base_dir, scene_id, mask_group_name, color_shape):
    mask_dir = join(base_dir, '2D_masks', scene_id, mask_group_name)
    color_name = basename(color_path)
    mask_name = 'maskraw_' + color_name.replace('jpg', 'png')
    mask_path = join(mask_dir, mask_name)
    mask_raw = cv2.imread(mask_path, -1).astype(np.float32)
    mask_raw = cv2.resize(
        mask_raw, [color_shape[1], color_shape[0]], interpolation=cv2.INTER_NEAREST)
    # print("mask:",mask_raw.shape)  #(968, 1296)
    return mask_raw


def scannetpp_get_semantic_mask(color_path, base_dir, scene_id, semantic_text, color_shape):
    mask_dir = join(base_dir, '2D_masks', scene_id, semantic_text)
    color_name = basename(color_path)
    # mask_name = 'maskraw_' + color_name.replace('jpg','png')
    mask_name = 'semantic_maskraw_' + color_name.replace('jpg', 'png')
    mask_path = join(mask_dir, mask_name)
    # print(mask_path)
    mask_raw = cv2.imread(mask_path, -1).astype(np.float32)
    mask_raw = cv2.resize(
        mask_raw, [color_shape[1], color_shape[0]], interpolation=cv2.INTER_NEAREST)
    # print("mask:",mask_raw.shape)  #(968, 1296)
    return mask_raw


"""
================================================================================================
postproecssing
================================================================================================
"""


def filter_by_population(points, filter_population, save_path):
    """filter out small groups of points
    """
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


def get_common_label(labels, max_label=None):
    """
    get the most common label from a group of labels (except for 0) 
    assign a new label when there exist three or more major label in a seg
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    if (unique_labels.shape[0] == 1):
        common_label = unique_labels[np.argsort(-counts)][0]
    else:
        fg_labels = unique_labels[unique_labels != 0]
        counts = counts[unique_labels != 0]
        prim_label_num = np.sum(counts > np.sum(counts)/counts.shape[0]*3)
        if (max_label is not None and prim_label_num >= 3):
            common_label = max_label + 1
        else:
            common_label = fg_labels[np.argsort(-counts)][0]

    return common_label


def get_seg_label(seg_ids, labels, points, voxel_size, max_label):
    new_labels = np.zeros(labels.shape[0])

    ids = np.unique(seg_ids)
    for id in ids:
        group = (seg_ids == id)
        group_points = points[group]
        group_labels = labels[group]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(group_points)
        pcd_ds, _, voxel_members = pcd.voxel_down_sample_and_trace(
            voxel_size,
            pcd.get_min_bound(),
            pcd.get_max_bound(),
            False)
        voxel_num = len(voxel_members)
        voxel_labels = np.zeros(voxel_num)

        # get labels of voxels
        for i in range(voxel_num):
            ids = voxel_members[i]
            voxel_labels[i] = get_common_label(group_labels[ids])

        seg_common_labels = get_common_label(voxel_labels, max_label)
        if (seg_common_labels > max_label):
            max_label = seg_common_labels
        new_labels[group] = seg_common_labels
    return new_labels


"""
================================================================================================
utils for evaluation
================================================================================================
"""


def export_ids(filename, ids):
    if (not os.path.exists(dirname(filename))):
        os.mkdir(dirname(filename))
    with open(filename, 'w') as f:
        for id in ids:
            f.write('%d\n' % id)


# standard scannet evaluation format
def export_res_to_class_agnostc_eval(base_dir, scene_id, pol_name, semantic_name=None):
    assert (pol_name.endswith('.pts'))
    pol_path = join(base_dir, 'scans', scene_id, 'results', pol_name)
    instance_ids = np.loadtxt(pol_path)[:, 4].astype(int)

    semantic_ids = np.ones_like(instance_ids, dtype=int)

    save_dir = join(base_dir, 'results', pol_name[:-4])
    os.makedirs(save_dir, exist_ok=True)

    filename = join(save_dir, f'{scene_id}.txt')
    output_mask_path_relative = f'{scene_id}_pred_mask'
    name = os.path.splitext(os.path.basename(filename))[0]
    output_mask_path = os.path.join(
        os.path.dirname(filename), output_mask_path_relative)
    if not os.path.isdir(output_mask_path):
        os.mkdir(output_mask_path)
    insts = np.unique(instance_ids)
    zero_mask = np.zeros(shape=(instance_ids.shape[0]), dtype=np.int32)
    with open(filename, 'w') as f:
        for idx, inst_id in tqdm(enumerate(insts)):
            if inst_id == 0:  # 0 -> no instance for this vertex
                continue
            relative_output_mask_file = os.path.join(
                output_mask_path_relative, name + '_' + str(idx) + '.txt')
            output_mask_file = os.path.join(
                output_mask_path, name + '_' + str(idx) + '.txt')
            loc = np.where(instance_ids == inst_id)
            label_id = semantic_ids[loc[0][0]]
            f.write('%s %d %f\n' % (relative_output_mask_file, label_id, 1.0))
            # write mask
            mask = np.copy(zero_mask)
            mask[loc[0]] = 1
            export_ids(output_mask_file, mask)


# just one txt for a scene, each line is a instance id
def new_export_res_to_class_agnostc_eval(base_dir, scene_id, pol_name, semantic_name=None):
    assert (pol_name.endswith('.pts'))
    pol_path = join(base_dir, 'scans', scene_id, 'results', pol_name)
    instance_ids = np.loadtxt(pol_path)[:, 4].astype(np.int32)

    # instance_ids = num_to_natural(instance_ids, void_number=0)
    print(np.unique(instance_ids), len(np.unique(instance_ids)))
    save_dir = join(base_dir, 'results', "new_formation_" + pol_name[:-4])
    os.makedirs(save_dir, exist_ok=True)

    np.savetxt(join(save_dir, f'{scene_id}.txt'), instance_ids.astype(np.int32), fmt='%d')
    print('save to ', join(save_dir, f'{scene_id}.txt'))


# export scannetpp gt into evaluation format 
def export_small_gt_to_class_agnostic_eval(base_dir, scene_id):
    gt_path = join(base_dir, 'scans', scene_id, 'scans', 'segments_anno.json')
    with open(gt_path, 'r') as f:
        data = json.load(f)

    seg_path = join(base_dir, 'scans', scene_id, 'scans', 'segments.json')
    with open(seg_path, 'r') as f:
        data2 = json.load(f)
    seg_ids = np.array(data2['segIndices'])

    classes_path = join(base_dir, 'metadata/semantic/instance_classes.txt')
    with open(classes_path, 'r') as f:
        classes = f.readlines()
    instance_valid_label = [valid_class.strip() for valid_class in classes]
    # print(instance_valid_label)

    eval_ids = np.zeros_like(seg_ids, dtype=int)
    vert_ins_size = np.ones_like(seg_ids, dtype=int)*1000000
    seg_groups = data['segGroups']
    for seg_group in seg_groups:
        segments = seg_group['segments']
        semantic_label = seg_group["label"]
        if (not semantic_label in instance_valid_label):
            continue
        inst_mask = np.isin(seg_ids, segments)
        inst_size = inst_mask.nonzero()[0].shape[0]
        inst_mask = inst_mask & (vert_ins_size > inst_size)

        eval_ids[inst_mask] = 1000 + seg_group['objectId'] + 1  # set valid class id to 1, means "everything"
        vert_ins_size[inst_mask] = inst_size

    os.makedirs(join(base_dir, 'results', 'small_class-agnostic_gt_ids'), exist_ok=True)
    np.savetxt(join(base_dir, 'results', 'small_class-agnostic_gt_ids',
                    f'{scene_id}.txt'), eval_ids, fmt='%d')
