import os, sys
import json

try:
    import numpy as np
except:
    print "Failed to import numpy package."
    sys.exit(-1)

try:
    from plyfile import PlyData, PlyElement
except:
    print "Please install the module 'plyfile' for PLY i/o, e.g."
    print "pip install plyfile"
    sys.exit(-1)

import util


# matrix: 4x4 np array
# points Nx3 np array
def transform_points(matrix, points):
    assert len(points.shape) == 2 and points.shape[1] == 3
    num_points = points.shape[0]
    p = np.concatenate([points, np.ones((num_points, 1))], axis=1)
    p = np.matmul(matrix, np.transpose(p))
    p = np.transpose(p)
    p[:,:3] /= p[:,3,None]
    return p[:,:3]


def export_ids(filename, ids):
    with open(filename, 'w') as f:
        for id in ids:
            f.write('%d\n' % id)


def load_ids(filename):
    ids = open(filename).read().splitlines()
    ids = np.array(ids, dtype=np.int64)
    return ids

def read_mesh_vertices(filename):
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
    return vertices


# export 3d instance labels for instance evaluation
def export_instance_ids_for_eval(filename, label_ids, instance_ids):
    assert label_ids.shape[0] == instance_ids.shape[0]
    scene_id = filename.split('/')[-1][:-4] #.../scene_id.txt
    output_mask_path_relative = scene_id + '_' + 'pred_mask'
    name = os.path.splitext(os.path.basename(filename))[0]
    output_mask_path = os.path.join(os.path.dirname(filename), output_mask_path_relative)
    if not os.path.isdir(output_mask_path):
        os.mkdir(output_mask_path)
    insts = np.unique(instance_ids)
    zero_mask = np.zeros(shape=(instance_ids.shape[0]), dtype=np.int32)
    with open(filename, 'w') as f:
        for idx, inst_id in enumerate(insts):
            if inst_id == 0:  # 0 -> no instance for this vertex
                continue
            relative_output_mask_file = os.path.join(output_mask_path_relative, name + '_' + str(idx) + '.txt')
            output_mask_file = os.path.join(output_mask_path, name + '_' + str(idx) + '.txt')
            loc = np.where(instance_ids == inst_id)
            label_id = label_ids[loc[0][0]]
            f.write('%s %d %f\n' % (relative_output_mask_file, label_id, 1.0))
            # write mask 
            mask = np.copy(zero_mask)
            mask[loc[0]] = 1
            export_ids(output_mask_file, mask)


# ------------ Instance Utils ------------ #

class Instance(object):
    instance_id = 0
    label_id = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, mesh_vert_instances, instance_id):
        if (instance_id == -1):
            return
        self.instance_id     = int(instance_id)
        self.label_id    = int(self.get_label_id(instance_id))
        self.vert_count = int(self.get_instance_verts(mesh_vert_instances, instance_id))

    def get_label_id(self, instance_id):
        return int(instance_id // 1000)

    def get_instance_verts(self, mesh_vert_instances, instance_id):
        return (mesh_vert_instances == instance_id).sum()

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_dict(self):
        dict = {}
        dict["instance_id"] = self.instance_id
        dict["label_id"]    = self.label_id
        dict["vert_count"]  = self.vert_count
        dict["med_dist"]    = self.med_dist
        dict["dist_conf"]   = self.dist_conf
        return dict

    def from_json(self, data):
        self.instance_id     = int(data["instance_id"])
        self.label_id        = int(data["label_id"])
        self.vert_count      = int(data["vert_count"])
        if ("med_dist" in data):
            self.med_dist    = float(data["med_dist"])
            self.dist_conf   = float(data["dist_conf"])

    def __str__(self):
        return "("+str(self.instance_id)+")"

def read_instance_prediction_file(filename, pred_path):
    lines = open(filename).read().splitlines()
    instance_info = {}
    abs_pred_path = os.path.abspath(pred_path)
    for line in lines:
        parts = line.split(' ')
        if len(parts) != 3:
            util.print_error('invalid instance prediction file. Expected (per line): [rel path prediction] [label id prediction] [confidence prediction]')
        if os.path.isabs(parts[0]):
            util.print_error('invalid instance prediction file. First entry in line must be a relative path')
        mask_file = os.path.join(os.path.dirname(filename), parts[0])
        mask_file = os.path.abspath(mask_file)
        # check that mask_file lives inside prediction path
        if os.path.commonprefix([mask_file, abs_pred_path]) != abs_pred_path:
            util.print_error('predicted mask {} in prediction text file {} points outside of prediction path.'.format(mask_file,filename))

        info            = {}
        info["label_id"] = int(float(parts[1]))
        info["conf"]    = float(parts[2])
        instance_info[mask_file]  = info
    return instance_info


def get_instances(ids, class_ids, class_labels, id2label):
    instances = {}
    for label in class_labels:
        instances[label] = []
    instance_ids = np.unique(ids)
    for id in instance_ids:
        if id == 0:
            continue
        inst = Instance(ids, id)
        if inst.label_id in class_ids:
            instances[id2label[inst.label_id]].append(inst.to_dict())
    return instances



def get_gt_filtered_instances(ids, filter_class_ids, class_labels, id2label):
    instances = {}
    for label in class_labels:
        instances[label] = []
    instance_ids = np.unique(ids)
    for id in instance_ids:
        if id == 0:
            continue
        inst = Instance(ids, id)
        if inst.label_id in filter_class_ids:
            inst.label_id = 1
            instances[id2label[inst.label_id]].append(inst.to_dict())
    return instances

'''
filter_class_ids: ids for filtering invalid instance
class_labels, id2label: another set of labels for evaluating, which has nothing to do with filter_class_ids. 
'''    
def get_gt_ids_and_instances(filename , gt_path, filter_class_ids, class_labels, id2label):
    instance_info = read_instance_prediction_file(filename , gt_path)
    for mask_path in instance_info:
        point_num = len(open(mask_path).read().splitlines())
        break
    # print(point_num)
    ids = np.zeros(point_num,dtype=int)
    instance_count = 0
    for mask_path in instance_info:
        # print(mask_path,instance_count)  not equal
        id = instance_info[mask_path]['label_id']*1000 + instance_count
        instance_count += 1  
        mask = ~(load_ids(mask_path) == 0)
        ids[mask] = id
    # print('ids:',np.unique(ids))
    
    return ids, get_gt_filtered_instances(ids, filter_class_ids, class_labels, id2label)

           
'''
filter_class_ids: ids for filtering invalid instance
class labels is set to [1,]
labels of all valid instances are set to 1
'''   
def get_class_agnostic_instances(ids, filter_class_ids, class_labels, id2label):
    instances = {}
    for label in class_labels:
        instances[label] = []
    instance_ids = np.unique(ids)
    for id in instance_ids:
        if id == 0:
            continue
        inst = Instance(ids, id)
        if inst.label_id in filter_class_ids:
            inst.label_id = 1
            instances[id2label[inst.label_id]].append(inst.to_dict())
    return instances

# cater for line 251(and following..) in script "evaluate_semantic_instance.py"
def save_scan_ids(scan_path,label_map_file,save_dir):
    
    def read_aggregation(filename):
        assert os.path.isfile(filename)
        object_id_to_segs = {}
        label_to_segs = {}
        with open(filename) as f:
            data = json.load(f)
            num_objects = len(data['segGroups'])
            for i in range(num_objects):
                object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
                label = data['segGroups'][i]['label']
                segs = data['segGroups'][i]['segments']
                object_id_to_segs[object_id] = segs
                if label in label_to_segs:
                    label_to_segs[label].extend(segs)
                else:
                    label_to_segs[label] = segs
        return object_id_to_segs, label_to_segs

    def read_segmentation(filename):
        assert os.path.isfile(filename)
        seg_to_verts = {}
        with open(filename) as f:
            data = json.load(f)
            num_verts = len(data['segIndices'])
            for i in range(num_verts):
                seg_id = data['segIndices'][i]
                if seg_id in seg_to_verts:
                    seg_to_verts[seg_id].append(i)
                else:
                    seg_to_verts[seg_id] = [i]
        return seg_to_verts, num_verts
    
    scan_name = os.path.split(scan_path)[-1]
    mesh_file = os.path.join(scan_path, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(scan_path, scan_name + '.aggregation.json')
    seg_file = os.path.join(scan_path, scan_name + '_vh_clean_2.0.010000.segs.json')
    label_map = util.read_label_mapping(label_map_file, label_from='raw_category', label_to='nyu40id')
    # print(scan_path,mesh_file)
    mesh_vertices = read_mesh_vertices(mesh_file)
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32) 
        
    for label, segs in label_to_segs.iteritems():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
            
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # id = [NYUv2 40-label]*1000+instance_id
    for object_id, segs in object_id_to_segs.iteritems():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id + label_ids[verts] * 1000
    
    save_path = os.path.join(save_dir, scan_name+'.txt')
    # print('save to ',save_path)
    export_ids(save_path, instance_ids)

# process geometry seg into instance eval format
def export_seg_ids_for_eval(scan_path,save_dir):
    scan_name = os.path.split(scan_path)[-1]
    scene_seg_path = os.path.join(scan_path, scan_name + '_vh_clean_2.0.010000.segs.json')
    with open(scene_seg_path,'r') as f: 
        seg_data = json.load(f)
    instance_ids = np.array(seg_data['segIndices'])
    label_ids = np.ones_like(instance_ids,dtype=int)  #set label id to 1
    
    filename = os.path.join(save_dir,scan_name+'.txt')
    output_mask_path_relative = scan_name + '_pred_mask'
    name = os.path.splitext(os.path.basename(filename))[0]
    output_mask_path = os.path.join(os.path.dirname(filename), output_mask_path_relative)
    if not os.path.isdir(output_mask_path):
        os.mkdir(output_mask_path)
        
    insts,counts = np.unique(instance_ids,return_counts=True)
    # print(insts.shape[0],end=' ')
    insts = insts[counts >= 100]
    # print(insts.shape[0])
    zero_mask = np.zeros(shape=(instance_ids.shape[0]), dtype=np.int32)
    
    with open(filename, 'w') as f:
        for idx, inst_id in enumerate(insts):
            if inst_id == 0:  # 0 -> no instance for this vertex
                continue
            relative_output_mask_file = os.path.join(output_mask_path_relative, name + '_' + str(idx) + '.txt')
            output_mask_file = os.path.join(output_mask_path, name + '_' + str(idx) + '.txt')
            loc = np.where(instance_ids == inst_id)
            label_id = label_ids[loc[0][0]]
            f.write('%s %d %f\n' % (relative_output_mask_file, label_id, 1.0))
            # write mask 
            mask = np.copy(zero_mask)
            mask[loc[0]] = 1
            export_ids(output_mask_file, mask)
    

