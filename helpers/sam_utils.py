import numpy as np
import copy
from PIL import Image
from torchvision import transforms
import torch


def get_sam_by_iou(image, mask_generator):
    '''
    assign mask labels to every pixel of the image (code credit: SAM3D)
    '''
    masks = mask_generator.generate(image)
    group_ids = np.full(image.shape[1:], -1, dtype=int)
    sorted_masks = sorted(masks, key=(lambda x: x["predicted_iou"]))
    # sorted_masks = masks[::-1]
    num_masks = len(masks)
    group_counter = 0
    for i in range(num_masks):
        group_ids[sorted_masks[i]["segmentation"]] = group_counter
        # print(sorted_masks[i]["predicted_iou"])
        group_counter += 1
    print(group_counter)
    return group_ids


def get_sam_by_area(image, mask_generator, reverse=True):
    masks = mask_generator.generate(image)
    # group_ids = np.full((image.shape[0], image.shape[1]), -1, dtype=int)
    group_ids = np.full(image.shape[1:], -1, dtype=int)
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=reverse)

    group_counter = 0
    for mask in sorted_masks:
        group_ids[mask["segmentation"]] = group_counter  # area从高到低
        group_counter += 1
    print(group_counter)
    return group_ids


def num_to_natural(group_ids):
    '''
    Change the group number to natural number arrangement (code credit: SAM3D)
    '''
    if np.all(group_ids == -1):
        return group_ids
    array = copy.deepcopy(group_ids).astype(int)

    unique_values = np.unique(array[array != -1])
    mapping = np.full(np.max(unique_values) + 2, -1)
    mapping[unique_values + 1] = np.arange(len(unique_values))  # map ith(start from 0) group_id to i
    array = mapping[array + 1]
    return array


def viz_mask(group_ids):
    array = np.zeros(tuple(group_ids.shape) + (3,))
    # print(array.shape)
    if np.all(group_ids == -1):
        return array
    unique_values = np.unique(group_ids[group_ids != -1])
    # print(unique_values)    
    for i in unique_values:
        array[group_ids == i] = np.random.random((3))

    # print(array)
    return array * 255


def my_prepare_image(image_pth):
    """
    apply transformation to the image. crop the image ot 640 short edge by default
    """
    image = Image.open(image_pth).convert('RGB')

    image_ori = np.asarray(image)
    images = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()

    return image_ori, images
