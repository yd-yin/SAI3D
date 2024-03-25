import open3d as o3d
import numpy as np
import os
from os.path import join
import tqdm
from sai3d_utils import get_points_from_ply

def export_mesh(name, v, f, c=None):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    if c is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(c)
    o3d.io.write_triangle_mesh(name, mesh)


def get_labels_from_eval_format(scene_id, res_dir):
    """Get instance id of each point from the evaluation format of ScanNet
    When the instance id is 0, it means the point is not in any instance.
    If one point is in multiple instances, the instance is the one with highest confidence.
    """
    masks_data_path = join(res_dir,f'{scene_id}.txt')
    labels = None
    with open(masks_data_path,'r') as f:
        label_id = 1
        for line in tqdm.tqdm(reversed(f.readlines()), desc='get instance ids from eval format'):
            rel_mask_path = line.split(' ')[0]
            mask_path = join(res_dir,rel_mask_path)
            ids = np.array(open(mask_path).read().splitlines(),dtype=np.int64)
            if labels is None:
                labels = np.zeros_like(ids)
            labels[ids > 0] = label_id
            label_id += 1
    return labels


def save_scannet_eval_format_to_mesh(scene_id, 
                                   res_dir,
                                   data_dir ='../data/ScanNet'):
    """Save the class-agnostic instance segmentation results 
    from ScanNet eval format into mesh for visualization.
    We assign random colors to each object label. 
    
    Args:
        scene_id: the id of scece to visualize 
        res_dir: the directory of the results for evaluation, 
                 like "/home/data/ScanNet/results/demo_scannet_..depth2"
        data_dir: the directory of the ScanNet dataset
    """
    labels = get_labels_from_eval_format(scene_id, res_dir)
    points_num = labels.shape[0]
    
    colors = np.ones((points_num,3))
    for label in np.unique(labels):
        if(label == 0):continue
        colors[labels == label] = np.random.rand(3) 
    
    ply_path = join(data_dir, 'scans', scene_id, f'{scene_id}_vh_clean_2.ply')
    mesh = o3d.io.read_triangle_mesh(ply_path)
    v = np.array(mesh.vertices)
    f = np.array(mesh.triangles)

    c_label = colors

    os.makedirs(join(data_dir, 'vis_mesh'), exist_ok=True)
    save_path = join(data_dir, 'vis_mesh', f'{scene_id}.ply')
    export_mesh(save_path, v, f, c_label)
    print('save to', save_path)
    
if __name__ == '__main__':
    scene_id = 'scene0011_00'
    res_dir = "/home/code/SAI3D/data/ScanNet/results/demo_scannet_5view_merge200_2-norm_semantic-sam_connect(0.9,0.5,5)_depth2"
    data_dir = "/home/code//SAI3D/data/ScanNet"
    save_scannet_eval_format_to_mesh(scene_id, res_dir, data_dir)