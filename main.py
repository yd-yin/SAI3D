import sys
import os
from os.path import join, dirname, abspath
import numpy as np
import glob
import argparse
from tqdm import tqdm
import scipy
from natsort import natsorted
import utils
from linetimer import CodeTimer

n_workers = 20


class AISeg:
    def __init__(self, points, args):
        """
        :param points_objness: (x, y, z, objectness)
        """
        self.points = points
        self.N = points.shape[0]


    def init_data(self, text, thres_gsam, scene_id, base_dir):
        # get image data
        self.poses, self.intrinsics, self.masks, self.depths = self.get_data(base_dir, scene_id, text, thres_gsam)
        self.M = self.masks.shape[0]
        self.H, self.W = self.masks.shape[-2:]


    def get_objness(self, thres_obj, objness=None, batch=None, n_neigh=None):
        if objness is not None:
            print('load pre-computed objness')
        elif batch is not None and n_neigh is not None:
            # objness = self.get_objectness_forward()
            objness = self.get_objness_unproject(batch, n_neigh)
        else:
            raise ValueError('init objness either from input, or by computation')

        # points with objects
        self.obj_flag = objness > thres_obj
        self.points_obj = self.points[self.obj_flag]     # (N_obj, 3)
        self.N_obj = self.points_obj.shape[0]

        return objness


    def get_objness_forward(self, ):
        """
        forward project N points to M images to get objectness
        it is also slow
        """
        # project N points to M images
        pts_cam, pixes = utils.parallel_world2cam_pixel(self.points, self.intrinsics, self.poses)  # (N, M, 3), (N, M, 2)

        p_cam0, pix0 = pts_cam, pixes  # (N, M, 3), (N, M, 2)
        w0, h0 = np.split(pix0, 2, axis=-1)
        w0, h0 = w0[..., 0], h0[..., 0]  # (N, M)
        bounded_flag0 = (0 <= w0) * (w0 <= self.W - 1) * (0 <= h0) * (h0 <= self.H - 1)  # (N, M)
        """
        Note that results from invalid indices are meaningless. 
        However, we clip invalid indices and also query the results in order to obtain a regular (N, M) array.
        When return the results, the validity must be considered.
        """
        label0 = self.masks[
            np.arange(self.M), h0.clip(0, self.H - 1), w0.clip(0, self.W - 1)]  # (N, M), querying labels from masks (M, H, W) by h (N, M) and w (N, M)

        # visibility
        real_depth0 = p_cam0[..., -1]  # (N, M)
        capture_depth0 = self.depths[np.arange(self.M), h0.clip(0, self.H - 1), w0.clip(0, self.W - 1)]  # (N, M), querying depths
        visible_flag0 = np.isclose(real_depth0, capture_depth0, rtol=0.15)

        valid_flag0 = visible_flag0 * bounded_flag0  # (N, M)

        f_seen = valid_flag0  # (N, M)
        f_obj = f_seen * (label0 != 0)  # (N, M)

        seen = f_seen.sum(axis=-1)  # (N, M), -> (N, )
        obj = f_obj.sum(axis=-1)  # (N, M) -> (N, )

        objness = np.zeros(self.N, np.float32)
        never_seen = seen == 0
        print(f'never seen: {never_seen.sum() / self.N:.3f}')
        objness[~never_seen] = obj[~never_seen] / seen[~never_seen]  # (N, )

        return objness


    def get_objness_unproject(self, batch, n_neigh):
        """
        get M/b (b, N, 2) batch_objness2, and sum to objness2 (N, 2), and get objness = obj / seen (N, )
        :return: objness (N, )
        """
        kdtree_points = scipy.spatial.KDTree(self.points)
        objness2 = np.zeros((self.N, 2), np.float32)

        # batch_objness2_1 = self.objectness_one_batch(kdtree_points, n_neigh, self.depths[[0]], self.intrinsics[[0]], self.poses[[0]],
        #                                              self.masks[[0]])  # (b, N, 2)

        for i in tqdm(range(np.ceil(self.M / batch).astype(int)), desc='Get objness'):
            sli = slice(i*batch, (i+1)*batch)
            batch_objness2 = self.objness_unproject_one_batch(kdtree_points, n_neigh, self.depths[sli], self.intrinsics[sli], self.poses[sli], self.masks[sli])     # (b, N, 2)
            objness2 += batch_objness2.sum(0)

        objness = np.zeros(self.N, np.float32)
        never_seen = objness2[:, -1] == 0
        print(f'never seen: {never_seen.sum() / self.N:.3f}')
        objness[~never_seen] = objness2[~never_seen, 0] / objness2[~never_seen, 1]      # (N, )

        return objness


    def objness_unproject_one_batch(self, kdtree, n_neigh, depths, intrinsics, poses, maskraws):
        """
        :param kdtree: kdtree_points
        :param n_neigh: number of neighbors in unprojection
        :param depths: (b, H, W)
        :param intrinsics: (b, 3, 3)
        :param poses: (b, 4, 4)
        :param maskraws: (b, H, W)
        :return: batch_objness2 (b, N, 2)
        """
        batch = depths.shape[0]

        batch_objness2 = np.zeros((batch, self.N, 2), np.float32)   # (b, N, 2), (obj, seen)

        points_camera = utils.batch_pixel2camera(intrinsics, depths)      # (b, H*W, 3)
        points_world = utils.batch_camera2world(points_camera, poses)     # (b, H*W, 3)
        points_world[(points_camera == [0, 0, 0]).all(axis=-1)] = 0     # filter out invalid points
        points_label = maskraws.reshape(batch, -1)               # (b, H*W)

        # seen points
        seen_id = kdtree.query(points_world, n_neigh, workers=n_workers)[1]     # (b, H*W, n_neigh)
        batch_objness2[np.arange(batch)[:, None], seen_id.reshape(batch, -1), 1] += 1  # seen

        # obj points
        foreground_mask = points_label != 0         # (b, H*W)
        # For foreground points, there is no way to form a (b, #foreground, 3) array, because #foreground for each image is different.
        # And also, there is no way to form a (b, #foreground, n_neigh) foreground_id array.
        for i in range(batch):
            obj_id_i = seen_id[i, foreground_mask[i]]       # (#foreground, n_neigh)
            batch_objness2[i, obj_id_i.reshape(-1), 0] += 1

        return batch_objness2



    def assign_label(self, thres_connect, coarse_radius=0.2, k_graph=8):
        # coarse label
        labels_obj_fine = np.zeros(self.N_obj, np.float32)      # (N_obj, )
        labels_coarse, labels_obj_coarse = self.assign_coarse_label(coarse_radius)      # (N, ), (N_obj, )

        if thres_connect == 0:
            return labels_coarse, labels_coarse

        # fine label
        labels_fine = np.zeros(self.N, np.float32)      # (N, )
        points_region_objId_list = self.split_region_by_label(labels_obj_coarse)

        for points_region_objId in tqdm(points_region_objId_list, desc='Assign fine labels'):
            if points_region_objId.size == 1:
                # a region only has 1 point, invalid
                continue
            points_region = self.points_obj[points_region_objId]
            labels_region_fine = self.region_assign_fine_label(points_region, thres_connect, k_graph)       # (N_region, )

            labels_obj_fine[points_region_objId] = labels_region_fine

        labels_fine[self.obj_flag] = labels_obj_fine

        return labels_coarse, labels_fine


    def region_assign_fine_label(self, points_region, thres_connect, k_graph):
        """
        Return labels_region_fine, this labels should be first plugged into labels_obj_fine (N_obj, ) by points_region_objId,
        and then labels_obj_fine should be plugged into labels_fine (N, ) by self.obj_flag
        :return: labels_region_fine, (N_region, )
        """
        N_region = points_region.shape[0]
        adj_region, neighID_mat = self.get_adjacency(points_region, k_graph=k_graph)

        labels_region_fine = np.zeros(N_region, np.float32)

        assign_id = 1
        for i in range(N_region):
            if labels_region_fine[i] == 0:
                queue = []
                queue.append(i)
                labels_region_fine[i] = assign_id

                while queue:
                    v = queue.pop(0)
                    js = neighID_mat[v]
                    for j in js:
                        if not labels_region_fine[j] == 0:
                            continue
                        connect = self.connect_judge_adj(adj_region, v, j, thres_connect)
                        if not connect:
                            continue
                        queue.append(j)
                        labels_region_fine[j] = assign_id

                assign_id += 1

        return labels_region_fine       # (N_region, )


    def assign_coarse_label(self, coarse_radius):
        labels_obj_coarse = np.zeros(self.N_obj, np.float32)   # labels_object_coarse  (N_obj, )

        kdtree_obj = scipy.spatial.KDTree(self.points_obj)
        assign_id = 1
        for i in tqdm(range(self.N_obj), desc='Assign coarse labels'):
            if labels_obj_coarse[i] == 0:
                queue = []
                queue.append(i)
                labels_obj_coarse[i] = assign_id

                while queue:
                    v = queue.pop(0)
                    js = self.get_neighbor_id_from_radius(kdtree_obj, self.points_obj[v], radius=coarse_radius)
                    need_assign_flag = labels_obj_coarse[js] == 0
                    if not need_assign_flag.any():
                        continue

                    js = js[need_assign_flag]
                    queue.append(js)
                    labels_obj_coarse[js] = assign_id

                assign_id += 1

        labels_coarse = np.zeros(self.N, np.float32)
        labels_coarse[self.obj_flag] = labels_obj_coarse

        return labels_coarse, labels_obj_coarse        # (N, ), (N_obj, )


    # ====== about adjacency ======
    def get_adjacency(self, points_any, k_graph):
        """
        :param points_any: points of one region with the same coarse labels. (N, 3), N can be N, N_obj, N_region, ...
        :param k_graph: build knn graph (N, k)
        :return: adjacency_mat, (N, N)
        """
        k_graph = min(k_graph, points_any.shape[0])
        kdtree = scipy.spatial.KDTree(points_any)

        # Containing the neighboring information, neighID_mat[i] contains the neighbor indices of i
        neighID_mat = kdtree.query(points_any, k_graph, workers=n_workers)[1]  # (N, k),

        # project N points to M images
        pts_cam, pixes = utils.parallel_world2cam_pixel(points_any, self.intrinsics, self.poses)  # (N, M, 3), (N, M, 2)

        connect_mat, seen_mat = self.get_connect_seen_mat(pts_cam, pixes, neighID_mat)  # (N, k)

        adjacency_mat = self.get_adjacency_from_score(neighID_mat, connect_mat, seen_mat)  # (N, N)

        return adjacency_mat, neighID_mat


    def get_connect_seen_mat(self, pts_cam, pixes, neighID_mat, visibility=True):
        """
        the first column is the original points_obj (the latest neighbor of points_obj), the remaining 7 columns are neighbors
        :param pts_cam: (N, M, 3), transformed to camera coordinate
        :param pixes: (N, M, 2), projected pixel locations
        :param neighID_mat: (N, k), the indices matrix from knn
        :return: connect: (N, k), seen: (N, k)
        """
        p_cam0, pix0 = pts_cam, pixes  # (N, M, 3), (N, M, 2)
        w0, h0 = np.split(pix0, 2, axis=-1)
        w0, h0 = w0[..., 0], h0[..., 0]  # (N, M)
        bounded_flag0 = (0 <= w0) * (w0 <= self.W - 1) * (0 <= h0) * (h0 <= self.H - 1)  # (N, M)
        """
        Note that results from invalid indices are meaningless. 
        However, we clip invalid indices and also query the results in order to obtain a regular (N, M) array.
        When return the results, the validity must be considered.
        """
        label0 = self.masks[
            np.arange(self.M), h0.clip(0, self.H - 1), w0.clip(0, self.W - 1)]  # (N, M), querying labels from masks (M, H, W) by h (N, M) and w (N, M)

        if visibility:
            real_depth0 = p_cam0[..., -1]  # (N, M)
            capture_depth0 = self.depths[np.arange(self.M), h0.clip(0, self.H - 1), w0.clip(0, self.W - 1)]  # (N, M), querying depths
            visible_flag0 = np.isclose(real_depth0, capture_depth0, rtol=0.15)

        labels_allk = label0[neighID_mat]  # (N, k, M)
        bounded_flag_allk = bounded_flag0[neighID_mat]  # (N, k, M)
        if visibility:
            visible_flag_allk = visible_flag0[neighID_mat]  # (N, k, M)

        connect_mat = []
        seen_mat = []

        for i in range(labels_allk.shape[1]):
            bounded_flag = bounded_flag0 * bounded_flag_allk[:, i]  # (N, M)

            label1 = labels_allk[:, i]

            if not visibility:
                valid_flag = bounded_flag  # (N, M)

            else:
                visible_flag = visible_flag0 * visible_flag_allk[:, i]  # (N, M)
                valid_flag = visible_flag * bounded_flag  # (N, M)

            f_both0 = valid_flag * ((label0 == 0) * (label1 == 0))  # (N, M)
            f_seen = valid_flag * ~f_both0  # (N, M)
            f_connect = f_seen * (label0 == label1)  # (N, M)

            connect = f_connect.sum(axis=-1)  # (N, M) -> (N, )
            seen = f_seen.sum(axis=-1)  # (N, M), -> (N, )

            connect_mat.append(connect)
            seen_mat.append(seen)

        connect_mat = np.stack(connect_mat, axis=-1)  # (N, k)
        seen_mat = np.stack(seen_mat, axis=-1)
        return connect_mat, seen_mat


    def get_adjacency_from_score(self, neighID_mat, connect_mat, seen_mat):
        """
        :param neighID_mat: (N, k)
        :param connect_mat: (N, k)
        :param seen_mat: (N, k)
        :return: adj: (N, N)
        """
        N_any, k_graph = neighID_mat.shape
        rows_ori = neighID_mat[:, 0]  # (N, )
        # Note that the operations follow the row-major order
        rows = np.repeat(rows_ori, k_graph)  # (N*k, ), (0,0,0,0,1,1,1,1,...)
        cols = neighID_mat.flatten()  # (N*k, ), (a0, a1, a2, a3, b0, b1, b2, b3, ...)
        connects = connect_mat.flatten()  # (N*k, )
        seens = seen_mat.flatten()  # (N*k, )

        adj_connect = scipy.sparse.coo_array((connects, (rows, cols)), shape=(N_any, N_any), dtype=np.float32).todok()  # (N, N)
        adj_seen = scipy.sparse.coo_array((seens, (rows, cols)), shape=(N_any, N_any), dtype=np.float32).todok()  # (N, N)
        if adj_seen.nonzero()[0].size == 0:
            # print("Nothing seen. Return adj=zeros. Probably you use the wrong gsam directory, i.e., wrong `text` and `thres_gsam`.")
            return adj_seen
        # assert adj_seen.nonzero()[0].size, "Nothing seen. Probably you use the wrong gsam directory, i.e., wrong `text` and `thres_gsam`."
        adj = scipy.sparse.dok_matrix(adj_seen.shape, dtype=np.float32)
        adj[adj_seen.nonzero()] = adj_connect[adj_seen.nonzero()] / adj_seen[adj_seen.nonzero()]

        # Sometimes, j is a neighbor of i (adj[i,j] = score), but i is not a neighbor of j (adj[j, i] = 0)
        # make sure adj is a symmetric matrix, by assign a[i,j] = a[j,i] = max(a[i,j], a[j,i])
        adj = adj.tocsr()
        r, c = adj.nonzero()
        adj[r, c] = adj[c, r] = np.maximum(adj[r, c], adj[c, r])

        return adj
    # ================================


    def connect_judge_adj(self, adj, p1_id, p2_id, thres_connect):
        score = adj[p1_id, p2_id]
        return score >= thres_connect


    def get_neighbor_id_from_radius(self, kdtree, p, radius):
        """
        Take one point or a batch of points, find the nearest neighbors by radius
        :param p: (3, ) or (b, 3)
        :param radius: in meters
        """
        p = p.reshape(-1, 3)
        neighbor_id = kdtree.query_ball_point(p, radius, workers=n_workers)
        neighbor_id = np.unique(np.concatenate(neighbor_id))
        return neighbor_id


    def split_region_by_label(self, labels_coarse):
        id_list = []
        for i in np.unique(labels_coarse):
            if i == 0:
                continue
            id_list.append(np.where(labels_coarse == i)[0])
        return id_list


    def get_data(self, base_dir, scene_id, text, thres_gsam):
        color_list = natsorted(glob.glob(join(base_dir, 'matterport_2d', scene_id, 'color', '*.jpg')))

        poses = []
        intrinsics = []
        masks = []
        depths = []

        for color_path in tqdm(color_list, desc='Read 2D data'):
            poses.append(utils.get_pose(color_path))
            intrinsics.append(utils.get_intrinsic(color_path))
            masks.append(utils.get_mask(color_path, text, thres_gsam))
            depths.append(utils.get_depth(color_path))

        poses = np.stack(poses, 0)  # (M, 4, 4)
        intrinsics = np.stack(intrinsics, 0)  # (M, 3, 3)
        masks = np.stack(masks, 0)  # (M, H, W)
        depths = np.stack(depths, 0)  # (M, H, W)

        return poses, intrinsics, masks, depths





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='data')
    parser.add_argument('--scene_id', type=str, default='RPmz2sHmrrY')
    parser.add_argument('--text', type=str, help='prompt')

    parser.add_argument('--thres_gsam', type=float, default=0.5)
    parser.add_argument('--thres_obj', type=float, default=0.3)
    parser.add_argument('--thres_connect', type=float, default=0.5)

    args = parser.parse_args()


    time_collection = {}

    with CodeTimer('Load points', dict_collect=time_collection):
        points_path = join(args.base_dir, 'matterport_2d', args.scene_id, 'points.pts')
        if not os.path.exists(points_path):
            utils.get_points_from_openscene_pth(args.base_dir, args.scene_id)
        save_dir = join(args.base_dir, 'matterport_2d', args.scene_id, 'results', args.text)
        points = np.loadtxt(points_path).astype(np.float32)

    agent = AISeg(points, args)

    with CodeTimer('Load images', dict_collect=time_collection):
        agent.init_data(args.text, args.thres_gsam, args.scene_id, args.base_dir)

    # 1. get objness
    with CodeTimer('Get objness', dict_collect=time_collection):
        objness_path = join(save_dir, f'objness_gsam{args.thres_gsam}.txt')
        if os.path.exists(objness_path):
            # load objness
            objness = agent.get_objness(thres_obj=args.thres_obj, objness=np.loadtxt(objness_path))
        else:
            # compute objness
            objness = agent.get_objness(thres_obj=args.thres_obj, batch=10, n_neigh=4)
            np.savetxt(objness_path, objness)
            print(f'save to {objness_path}')


    # 2. assign coarse and fine instance labels
    with CodeTimer('Assign instance labels', dict_collect=time_collection):
        labels_coarse, labels_fine = agent.assign_label(args.thres_connect)
        # get global fine labels by coarse labels and local fine labels
        labels_fine_global = np.unique(np.stack((labels_coarse, labels_fine), 1), axis=0, return_inverse=True)[1]


    with CodeTimer('Save results', dict_collect=time_collection):
        points_objness_label = np.concatenate((points, objness[:, None], labels_fine_global[:, None]), -1)        # (x, y, z, objectness, labels_fine)
        save_name = f'points_objness_label_gsam{args.thres_gsam}_obj{args.thres_obj}_connect{args.thres_connect}.pts'
        save_path = join(save_dir, save_name)
        np.savetxt(save_path, points_objness_label)
        print(f'save to {save_path}')

    # post-processing
    utils.filter_by_population(points_objness_label, (50, 100, ), save_path)


    for k, v in time_collection.items():
        print(f'Time {k}: {v:.1f}')
    print(f'Total time: {sum(time_collection.values()):.1f}')

