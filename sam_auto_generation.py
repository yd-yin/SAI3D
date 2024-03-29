import os
import tqdm
import cv2
import glob
import argparse
from natsort import natsorted
from helpers.sam_utils import get_sam_by_iou, get_sam_by_area, num_to_natural, viz_mask, my_prepare_image
from semantic_sam import build_semantic_sam, SemanticSamAutomaticMaskGenerator


def seg_scannet(base_dir, view_freq):

    with open(os.path.join(base_dir, 'Tasks/Benchmark/scannetv2_val.txt'), 'r') as f:
        val_split = f.readlines()
    val_split = [s.strip() for s in val_split]
    seg_split = sorted(val_split)

    all_color_base = os.path.join(base_dir, 'posed_images')

    level = [3,]  # instance level

    sam_model = build_semantic_sam(
            model_type='L', ckpt='checkpoints/swinl_only_sam_many2many.pth')
    mask_generator = SemanticSamAutomaticMaskGenerator(
        sam_model,
        level=level)  # model_type: 'L' / 'T', depends on your checkpoint

    os.makedirs(os.path.join(base_dir, '2D_masks'), exist_ok=True)
    for scene_id in tqdm.tqdm(seg_split):
        color_base = os.path.join(all_color_base, scene_id)
        color_paths = natsorted(glob.glob(os.path.join(color_base, '*.jpg')))
        for color_path in tqdm.tqdm(color_paths, desc=scene_id):
            color_name = os.path.basename(color_path)
            num = int(color_name[-9:-4])
            if num % view_freq != 0:
                continue
            print(color_path)
            original_image, input_image = my_prepare_image(image_pth=color_path)
            labels = get_sam_by_iou(input_image, mask_generator)
            # labels = get_sam_by_area(input_image,mask_generator)
            color_mask = viz_mask(labels)
            labels = num_to_natural(labels) + 1  # 0 is background

            save_path = os.path.join(base_dir, '2D_masks', scene_id)
            if (not os.path.exists(save_path)):
                os.mkdir(save_path)
            save_path = os.path.join(save_path, 'semantic-sam')
            if (not os.path.exists(save_path)):
                os.mkdir(save_path)
            # cv2.imwrite(os.path.join(save_path,color_name),original_image)
            cv2.imwrite(os.path.join(
                save_path, f'maskcolor_{color_name[:-4]}.png'), color_mask)
            cv2.imwrite(os.path.join(
                save_path, f'maskraw_{color_name[:-4]}.png'), labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--view_freq', type=int, default=5,
                        help='sample freuqncy for views')
    args = parser.parse_args()

    seg_scannet(base_dir=args.data_dir, view_freq=args.view_freq)
