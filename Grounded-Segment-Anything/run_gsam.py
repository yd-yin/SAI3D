import argparse
import os
import time
from os.path import join, basename, dirname, abspath, splitext
import copy
import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor


device = 'cuda'


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(args):
    args_dino = SLConfig.fromfile(args.config)
    model = build_model(args_dino)
    checkpoint = torch.load(args.grounded_checkpoint, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (900, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (900, 4)
    logits_max = logits.max(dim=1)[0]
    """
    For each query: 
    The model outputs the logits (per token classification score), and the box location
    if max(logits) > box_threshold: the query is preserved
    predicted_text = decode(tokens with score > text_threshold)
    """


    # filter output
    filt_mask = logits_max > box_threshold       # (900, ), where True denotes a valid box
    score_filt = logits_max.clone()[filt_mask]     # (num_filt, )

    logits_filt = logits.clone()[filt_mask]  # (num_filt, 256)
    boxes_filt = boxes.clone()[filt_mask]  # (num_filt, 4)

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, score, box in zip(logits_filt, score_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(score.item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)


    # sort the masks by its score, so that mask[0] < mask[1] < ... The latter masks will replace the former in mask_raw_png
    score_filt, order_id = score_filt.sort(descending=False)
    boxes_filt = boxes_filt[order_id]
    pred_phrases = np.array(pred_phrases)[order_id.numpy()].tolist()

    return boxes_filt, pred_phrases, score_filt


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, file_id, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(join(output_dir, f'mask_{file_id}.png'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, f'mask_{file_id}.json'), 'w') as f:
        json.dump(json_data, f)


def get_sam_output(predictor, image, boxes_filt):
    predictor.set_image(image)

    H, W, _ = image.shape
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )

    return masks



def run(image_path, DINO_model, SAM_model, args):
    # load image
    image_pil, image_tensor = load_image(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    file_id = splitext(basename(image_path))[0].replace('color_', '')
    gsam_save_path = join(args.output_dir, f'gsam_{file_id}.jpg')
    gsam_save_path_valid = join(args.output_dir_valid, f'gsam_{file_id}.jpg')
    # if os.path.exists(gsam_save_path):
    #     print(f'\nAlready exists, skip {gsam_save_path}\n')
    #     return

    # visualize raw image
    # image_pil.save(join(args.output_dir, f'color_{file_id}.png'))

    # run grounding dino model
    boxes_filt, pred_phrases, score_filt = get_grounding_output(DINO_model, image_tensor, args.text, args.box_threshold, args.text_threshold, device=device)

    if boxes_filt.shape[0] == 0:
        # print('\nNothing found, save raw image\n')
        # save raw image
        image_pil.save(gsam_save_path)
        # return
        masks = torch.zeros((0, 1, image.shape[0], image.shape[1]), dtype=torch.bool)
        valid = False

    else:
        # run sam model
        masks = get_sam_output(SAM_model, image, boxes_filt)
        valid = True

    # save raw masks as npy and png
    # save scores as png
    mask_raw_npy = masks[:, 0].cpu().numpy()
    np.save(join(args.output_dir, f'maskraw_{file_id}.npy'), mask_raw_npy)
    mask_raw_png = np.zeros_like(image[..., 0])
    score_png = np.zeros_like(image[..., 0]).astype(np.uint16)
    for i in range(mask_raw_npy.shape[0]):
        m = mask_raw_npy[i]
        m_true = np.argwhere(m)
        mask_raw_png[m_true[:, 0], m_true[:, 1]] = i + 1        # bugfix: the value can not be 0, otherwise it is seen as backgroud
        score_png[m_true[:, 0], m_true[:, 1]] = (score_filt[i].numpy() * 1000).round().astype(np.uint16)
    cv2.imwrite(join(args.output_dir, f'maskraw_{file_id}.png'), mask_raw_png)
    cv2.imwrite(join(args.output_dir, f'score_{file_id}.png'), score_png)

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    plt.axis('off')
    plt.savefig(
        gsam_save_path,
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
    if valid:
        plt.savefig(
            gsam_save_path_valid,
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )

    # save mask.jpg and metadata
    save_mask_data(args.output_dir, file_id, masks, boxes_filt, pred_phrases)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, help="path to config file", default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--grounded_checkpoint", type=str, help="path to checkpoint file", default="ckpt/groundingdino_swint_ogc.pth")
    parser.add_argument("--sam_checkpoint", type=str, help="path to checkpoint file", default="ckpt/sam_vit_h_4b8939.pth")
    parser.add_argument("--input_image", type=str, help="path to image file")
    parser.add_argument("--input_dir", type=str, help="directory to image file")
    parser.add_argument("--output_dir", type=str, help="output directory")

    parser.add_argument('--base_dir', type=str, default='../data')
    parser.add_argument('--scene_id', type=str, default='RPmz2sHmrrY')
    parser.add_argument("--text", type=str, help="text prompt")
    parser.add_argument("--thres", dest="box_threshold", type=float, default=0.5,
                        help="box threshold, boxes whose highest similarities higher than box_threshold are preserved")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                        help="text threshold, words whose similarities are higher than the text_threshold are chosen as predicted labels")
    parser.add_argument("--gpu", '-g', type=str, default='0')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not args.input_dir:
        args.input_dir = join(args.base_dir, 'matterport_2d', args.scene_id, 'color')

    if not args.output_dir:
        args.output_dir = join(args.base_dir, 'matterport_2d', args.scene_id, 'results', args.text, f'{args.text}_gsam{args.box_threshold:.2f}')
        print(f'set output_dir as {args.output_dir}\n')

    args.output_dir_valid = join(args.output_dir, 'valid')


    # ===== init ====
    # Grounded DINO
    DINO_model = load_model(args)
    # SAM
    SAM_model = SamPredictor(build_sam(checkpoint=args.sam_checkpoint).to(device))

    # make dir
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_valid, exist_ok=True)

    if args.input_image:
        run(args.input_image, DINO_model, SAM_model, args)
    else:
        image_list = sorted(glob.glob(join(args.input_dir, '*')))
        for image_path in tqdm(image_list):
            run(image_path, DINO_model, SAM_model, args)


# python run_gsam.py --text="picture frame" --scene_id=RPmz2sHmrrY --thres=0.5 -g=0
