#!/bin/bash
#SBATCH --account CIA-DAMTP-SL2-GPU
#SBATCH --partition ampere
#SBATCH -t 36:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

source ~/.bashrc
conda activate open-mmlab
cd /home/ll610/Onepiece/code_cs138/github/TrafficDataset/mmdetection/

# python tools/test.py configs/instaboost/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco.py work_dirs/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco/epoch_24.pth --eval segm
# python tools/test.py configs/instaboost/cascade_mask_rcnn_r101_fpn_instaboost_4x_coco.py work_dirs/cascade_mask_rcnn_r101_fpn_instaboost_4x_coco/epoch_24.pth --eval segm
# python tools/test.py configs/solov2/solov2_r50_fpn_3x_coco.py work_dirs/solov2_r50_fpn_3x_coco/epoch_24.pth --eval segm
# python tools/test.py configs/solov2/solov2_r101_fpn_3x_coco.py work_dirs/solov2_r101_fpn_3x_coco/epoch_24.pth --eval segm

# python tools/test.py configs/noisyboundaries/coco/mask_rcnn_r50_fpn_1x_coco_generatepl.py  work_dirs/noisyboundaries/coco/mask_rcnn_r50_fpn_1x_coco_sup/epoch_24.pth --out labels/unlabeled_1_1_coco.pkl
# python tools/test.py configs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_generatepl_1_2.py  work_dirs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_sup_1_2/epoch_24.pth --out labels/unlabeled_1_2_coco.pkl
# python tools/test.py configs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_generatepl_1_4.py  work_dirs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_sup_1_4/epoch_24.pth --out labels/unlabeled_1_4_coco.pkl
# python tools/test.py configs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_generatepl_1_8.py  work_dirs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_sup_1_8/epoch_24.pth --out labels/unlabeled_1_8_coco.pkl

python tools/test.py work_dirs/noisyboundaries/coco/mask_rcnn_r50_fpn_1x_coco_pl/mask_rcnn_r50_fpn_1x_coco_pl.py  work_dirs/noisyboundaries/coco/mask_rcnn_r50_fpn_1x_coco_pl/epoch_1.pth --eval segm