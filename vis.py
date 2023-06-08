import os
from os.path import join, basename
import numpy as np
import glob
import open3d as o3d
import matplotlib
from matplotlib import pyplot as plt
import argparse


def random_color(text, points_name):
    points_path = join(base_dir, 'matterport_2d', scene_id, 'results', text, points_name)
    points = np.loadtxt(points_path)
    labels = np.unique(points[..., -1])
    colors = np.zeros((points.shape[0], 3), np.float32)
    alphas = np.ones((points.shape[0], 1), np.float32)
    for l in labels:
        idx = points[..., -1] == l
        if l == 0:
            colors[idx] = 0.8 * np.ones(3, np.float32)
            alphas[idx] = 0.5
        else:
            colors[idx] = np.random.uniform(0, 0.8, size=3).astype(np.float32)
    points_color = np.concatenate((points[:, :3], colors, alphas), -1)
    save_path = points_path.replace('.pts', '_color.pts')
    np.savetxt(save_path, points_color)
    print('save to', save_path)


def heatmap(text, points_name):
    points_path = join(base_dir, 'matterport_2d', scene_id, 'results', text, points_name)
    points = np.loadtxt(points_path)
    objness = points[..., -1]

    objness[objness < 0.3] = 0

    m = matplotlib.cm.ScalarMappable(cmap='jet')
    # `to_rgba` returns [r, g, b, a] while we only want [r, g, b]
    colors = m.to_rgba(objness, bytes=False, norm=True)[:, :-1]
    colors[(colors == m.to_rgba(0, bytes=False, norm=True)[:-1]).all(1)] = 0.8 * np.ones(3, np.float32)

    points_color = np.concatenate((points[:, :3], colors), -1)
    save_path = points_path.replace('.pts', '_color.pts')
    np.savetxt(save_path, points_color)
    print('save to', save_path)



def export_mesh(name, v, f, c=None):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    if c is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(c)
    o3d.io.write_triangle_mesh(name, mesh)



def convert_color_to_mesh(text, points_color_name):
    scene_mesh_list = sorted(glob.glob(join(base_dir, 'matterport_2d', scene_id, 'region_segmentations', '*.ply')))
    work_dir = join(base_dir, 'matterport_2d', scene_id, 'results', text)
    points_color = np.loadtxt(join(work_dir, points_color_name))
    color_label = points_color[:, 3:6]


    start_idx = 0
    for scene_mesh_name in scene_mesh_list:
        mesh = o3d.io.read_triangle_mesh(scene_mesh_name)
        v = np.array(mesh.vertices)
        f = np.array(mesh.triangles)
        c = np.array(mesh.vertex_colors)

        c_label = color_label[start_idx: start_idx + v.shape[0]]
        start_idx += v.shape[0]

        os.makedirs(join(work_dir, 'vis_mesh'), exist_ok=True)
        save_path = join(work_dir, 'vis_mesh', basename(scene_mesh_name).replace('.ply', '_color.ply'))
        export_mesh(save_path, v, f, c_label)
        print('save to', save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='data/')
    parser.add_argument('--scene_id', type=str, default='RPmz2sHmrrY')
    parser.add_argument('--text', type=str, help='prompt')
    parser.add_argument('--pts_name', type=str, help='points_objness_label_name')

    args = parser.parse_args()

    base_dir = args.base_dir
    scene_id = args.scene_id
    text = args.text

    # objness
    # points_objness = 'points_objness_gsam0.5.pts'
    # heatmap(text, points_objness)
    # convert_color_to_mesh(text, points_objness.replace('.pts', '_color.pts'))

    # coarse label
    # points_coarse_name = 'points_coarse_gsam0.5.pts'
    # random_color(text, points_coarse_name)
    # convert_color_to_mesh(text, points_coarse_name.replace('.pts', '_color.pts'))

    # fine label
    # points_objness_label_name = 'points_objness_label_gsam0.5_obj0.3_connect0.3_amount300.pts'
    points_objness_label_name = args.pts_name
    random_color(text, points_objness_label_name)
    convert_color_to_mesh(text, points_objness_label_name.replace('.pts', '_color.pts'))

    # blender
    # work_dir = join(base_dir, 'matterport_2d', scene_id, 'results', text, 'vis_mesh')
    # cmd = f"/DATA1/yingda/software/blender-2.82-linux64/blender --background --python m_process_blender.py -- " \
    #       f"--work_dir=\"{work_dir}\" --scene_id={scene_id} --save_name=\"{text}_{points_objness_label_name.replace('.pts', '_color.pts')}\""
    # os.system(cmd)





