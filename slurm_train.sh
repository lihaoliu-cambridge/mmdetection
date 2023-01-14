#!/bin/bash
#SBATCH --account CIA-DAMTP-SL2-GPU
#SBATCH --partition ampere
#SBATCH -t 36:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

source ~/.bashrc
conda activate open-mmlab
cd /home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/

# python tools/train.py configs/instaboost/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco.py
# python tools/train.py configs/instaboost/cascade_mask_rcnn_r101_fpn_instaboost_4x_coco.py
# python tools/train.py configs/solov2/solov2_r50_fpn_3x_coco.py
# python tools/train.py configs/solov2/solov2_r101_fpn_3x_coco.py
# python tools/train.py configs/queryinst/queryinst_r50_fpn_mstrain_480-800_3x_coco.py
# python tools/train.py configs/queryinst/queryinst_r101_fpn_mstrain_480-800_3x_coco.py
# python tools/train.py configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py
# python tools/train.py configs/mask2former/mask2former_r101_lsj_8x2_50e_coco.py
# python tools/train.py configs/mask2former/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco.py
# python tools/train.py configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py



# python tools/train.py configs/noisyboundaries/coco/mask_rcnn_r50_fpn_1x_coco_sup.py
# python tools/train.py configs/noisyboundaries/coco/mask_rcnn_r50_fpn_1x_coco_pl.py

# python tools/train.py configs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_sup_1_2.py
# python tools/train.py configs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_pl_1_2.py

# python tools/train.py configs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_sup_1_4.py
# python tools/train.py configs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_pl_1_4.py

# python tools/train.py configs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_sup_1_8.py
python tools/train.py configs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_pl_1_8.py
