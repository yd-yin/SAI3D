# SAI3D: Segment Any Instance in 3D with Open Vocabularies

 [Yingda Yin<sup>1,2*</sup>](https://yd-yin.github.io/), [Yuzheng Liu<sup>2,3*</sup>](https://github.com/Ly-kc/) [Yang Xiao<sup>4*</sup>](https://youngxiao13.github.io/), [Daniel Cohen-Or<sup>5</sup>](https://danielcohenor.com/), [Jingwei Huang<sup>6</sup>](https://cs.stanford.edu/people/jingweih/), [Baoquan Chen<sup>2,3</sup>](http://cfcs.pku.edu.cn/baoquan/)

<sup>1</sup>School of Computer Science, Peking University &nbsp; &nbsp;
<sup>2</sup>National Key Lab of General AI, China &nbsp; &nbsp; 
<sup>3</sup>School of Intelligence Science and Technology, Peking University &nbsp; &nbsp; 
<sup>4</sup>Ecole des Ponts ParisTech&nbsp; &nbsp;           <sup>5</sup>Tel-Aviv University &nbsp; &nbsp;
<sup>6</sup>Tencent &nbsp; &nbsp;

**CVPR 2024**

[Project Page](https://yd-yin.github.io/SAI3D/) | [Arxiv Paper](https://arxiv.org/abs/2312.11557)

## Introduction

We introduce SAI3D, a novel zero-shot 3D instance segmentation approach that synergistically leverages geometric priors and semantic cues derived from Segment Anything Model (SAM). 

<img src="assets\pipeline.png" style="zoom: 33%;" />

Our approach combines geometric priors with the capabilities of 2D foundation models. We over-segment 3D point clouds into superpoints (top-left), and generate 2D image masks using SAM (bottom-left). We then construct a scene graph that quantifies the pairwise affinity scores of super points (middle). Finally, we leverage a progressive region growing to gradually merge 3D superpoints into the final 3D instance segmentation masks (right).

## Usage

### Installation

Prepare environment

```bash
conda create -n sai3d python=3.8
conda activate sai3d
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install open3d natsort matplotlib tqdm opencv-python scipy plyfile
```

Install Semantic-SAM

```bash
git clone https://github.com/UX-Decoder/Semantic-SAM.git Semantic-SAM --recursive
#if you encounter any problem about cuda version, try using cuda11.8 with the following command
#conda install nvidia/label/cuda-11.8.0::cuda  
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install git+https://github.com/cocodataset/panopticapi.git
cd Semantic_SAM
python -m pip install -r requirements.txt
cd semantic_sam/body/encoder/ops
sh ./make.sh
cd - && mkdir checkpoints && cd checkpoints
wget https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth
```

Install OpenMask3D(if need semantic)
```bash
git clone https://github.com/OpenMask3D/openmask3d.git openmask3d --recursive
cd openmask3d
conda create --name=openmask3d python=3.8.5 # create new virtual environment
conda activate openmask3d # activate it
bash install_requirements.sh  # install requirements
pip install -e .  # install current repository in editable mode
mkdir checkpoints && cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  #download SAM ckpt
```

### Data Preparation

#### ScanNet
Download [ScanNetV2 / ScanNet200](https://github.com/ScanNet/ScanNet) and organize the dataset as follows:
 
```
data
 ├── ScanNet
 │   ├── posed_images
 │   |   ├── scene0000_00
 │   |   │   ├──intrinsic_color.txt   
 │   |   │   ├──intrinsic_depth.txt   
 │   |   │   ├──0000.jpg     //rgb image
 │   |   │   ├──0000.png     //depth image
 │   |   │   ├──0000.txt     //extrinsic
 │   |   │   └── ...
 │   |   └── ...
 │   ├── scans
 │   |   ├── scene0000_00
 │   |   └── ...
 │   ├── Tasks
 │   |   ├── Benchmark
 │   |   │   ├──scannetv2_val.txt  
 │   |   │   ├──scannetv2_train.txt  
 │   |   │   └── ...
```



### Get class-agnostic masks

1. **Obtain 2D SAM results**
   
   Change [the config here](https://github.com/UX-Decoder/Semantic-SAM/blob/e3b9/configs/semantic_sam_only_sa-1b_swinL.yaml#L42) to false, and set the required parameter in this [script](scripts/sam_scannet.sh) then run:
   ```bash
   bash ./scripts/sam_scannet.sh
   ```

   The results will be stored at `data/ScanNet/2D_masks`, where the 2D segmentation results and visualization of 2D masks will be named as `maskraw_<frame_number>.png` and `maskcolor_<frame_number>.png` respectively.

2. **Obtain 3D superpoints**
   For ScanNet dataset, superpoints are already provided in `scans/<scene_id>/<scene_id>_vh_clean_2.0.010000.segs.json`

   To generate superpoint on mesh of other dataset, we also use the mesh segmentator provided by ScanNet directly. Please check [here](https://github.com/ScanNet/ScanNet/tree/master/Segmentator) to see the usage.


3. **3D instance segmentation by region growing**

   Set the required parameter in this [script](scripts/seg_scannet.sh), then run SAI3D by using the following command:
   
   ```bash
   bash scripts/seg_scannet.sh
   ```

   The resulting class-agnostic masks will be exported into the format for [ScanNet instance segmentation benchmark](https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py).



### Evaluate class-agnostic results
   Now you can implement class-agnostic evaluation directly on the results we got, which focuses only on the accuracy of the instance masks without considering any semantic label

   We modify the original ScanNet instance segmentation benchmark to conduct it. We collect all 18 classes(excluding wall and floor) of gt masks in ScanNet-v2 dataset as our gt class-agnostic masks, and the AP score is reported over all of the foreground masks. 

   We provide processed gt class-agnostic masks [here](https://drive.google.com/file/d/1VwDtNTCzPEbQaK7xmG6KXzHvAicKDIc_/view?usp=sharing). Please download and extract it into your `GT_DIR`

   1. Prepare environment for ScanNet benchmark
      ```bash
      conda create -n eval python=2.7
      conda activate eval
      cd evaluation
      pip install -r requirements.txt
      ```
   2. Start evaluation
      ```bash
      python evauation/evaluate_class_agnostic_instance.py \
      --pred_path=PREDICTION_DIR \
      --gt_path=GT_DIR
      ```

   The numerical results will be saved under the directory of your predictions by default.

### Visualize class-agnostic results
   Since the segmentation results in ScanNet evaluation format are tough to visualize, we provide functions in [helpers/visualize.py](helpers/visualize.py) to transform them into mesh(.ply) for visualization. Please check it to see the usage.

### Assign semantic with OpenMask3D and conduct 3D instance segmentation evaluation
   We prove that our proposed class-agnostic masks are more accurate and can be adopted in tasks like semantic instance segmentation. Here we choose OpenMask3D to assign semantic label for our class-agnostic masks.

   1. Reorganize scannet dataset 

      Since OpenMask3D requires ScanNet dataset to be organized like [this](https://github.com/OpenMask3D/openmask3d/blob/fb9b/README.md?plain=1#L148-L168), we provide a script to reorganize the dataset with softlink.  
      ```bash
         python helpers/format_convertion.py            \
         --app=0                                        \
         --base_dir=PATH_TO_PREVIOUS_SCANNET_DATASET    \
         --out_dir=PATH_TO_REORGANIZED_SCANNET_DATASET
      ```
      For example, 
      ```bash
         python helpers/format_convertion.py            \
         --app=0                                        \
         --base_dir="data/ScanNet"                      \
         --out_dir="data/ScanNet_OpenMask3D"
      ```
      > According to the convention of OpenMask3D, color and depth image of your data should share the same resolution. If not, please replace [this line in OpenMask3D](https://github.com/OpenMask3D/openmask3d/blob/6488/openmask3d/data/load.py#L73) with the following codes to adjust the resolution of color image to the same as depth image's when loading them in OpenMask3D:
      ```python
         img = Image.open(img_path).convert("RGB").resize(DEPTH_RESOLUTION,Image.BILINEAR)
         images.append(img)
      ```

   2. Prepare class-agnostic masks

      We've already got class-agnosic predictions from the previous section, and exported them into evaluation format for ScanNet benchmark.

      However, OpenMask3D requires class-agnostic masks to be saved in a `.pt` format before assigning semantic for them. So please run the following command to convert the previous format of class-agnostic predictions into the input format required by OpenMask3D. 

      ```bash
         python helpers/format_convertion.py  \ 
         --app=1                              \   
         --base_dir=PATH_TO_PREDICTION_DIR    \
         --out_dir=PATH_TO_SAVE_PREDICTION_OF_NEW_FORMAT
      ```
      For example,
      ```bash
         RESULT_NAME="demo_scannet_5view_merge200_2-norm_semantic-sam_connect(0.9,0.5,5)_depth2"
         python helpers/format_convertion.py                  \     
         --app=1                                              \
         --base_dir="data/ScanNet/results/${RESULT_NAME}"     \
         --out_dir="data/class_agnostic_masks"
      ```

   3. Assign semantic and evaluate
   
      We provide processed gt masks for ScanNet200 semantic instance segmentation [here](https://drive.google.com/file/d/1FYjzh6U8Em9BrKSw8f1OppgmeKtk1Ude/view?usp=sharing). 

      Now you can compute the per-mask scene features and run the evaluation of OpenMask3D on validation split of ScanNet200 dataset. Change the [intrinsic_resolution parameter in OpenMask3D configuration](https://github.com/OpenMask3D/openmask3d/blob/main/openmask3d/configs/openmask3d_scannet200_eval.yaml#L9) with the resolution of your `intrinsic_color.txt`. Then set the required parameter in this [script](scripts/run_openmask3d_scannet200.sh) and run the following command:
      
      ```bash
      bash scripts/run_openmask3d_scannet200.sh
      ```

      This script first computes the mask features associated with each class-agnostic mask, and then query masks with 200 class names in ScanNet200 to assign semantic label for them. Afterwards, the evaluation script automatically runs in order to obtain 3D closed-vocabulary semantic instance segmentation scores.
