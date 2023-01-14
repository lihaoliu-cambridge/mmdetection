
GPUS=$1
RESULTNAME=$2
ANNFILE=$3

#bash tools/dist_test.sh configs/noisyboundaries/coco/mask_rcnn_r50_fpn_1x_coco_generatepl.py  work_dirs/mask_rcnn_r50_fpn_1x_coco_sup/epoch_12.pth $GPUS --out $RESULTNAME

python scripts/coco/pkl2json.py $RESULTNAME

python scripts/coco/filter_pl.py $RESULTNAME

python scripts/coco/form_ann.py $RESULTNAME $ANNFILE










python tools/test.py configs/noisyboundaries/coco/mask_rcnn_r50_fpn_1x_coco_generatepl.py  work_dirs/noisyboundaries/coco/mask_rcnn_r50_fpn_1x_coco_sup/epoch_24.pth --out labels/unlabeled_1_1_coco.pkl
python scripts/coco/pkl2json.py  labels/unlabeled_1_1_coco.pkl --generatepl_config 'configs/noisyboundaries/coco/mask_rcnn_r50_fpn_1x_coco_generatepl.py'
python scripts/coco/filter_pl.py labels/unlabeled_1_1_coco.pkl --labeled_json 'data/coco/annotations/instances_train2017.1_1-labeled.json' --unlabeled_json 'data/coco/annotations/instances_train2017.1_1-unlabeled.json'
python scripts/coco/form_ann.py  labels/unlabeled_1_1_coco.pkl data/coco/annotations/instances_train2017.1_1-pl.json --labeled_json 'data/coco/annotations/instances_train2017.1_1-labeled.json' --unlabeled_json 'data/coco/annotations/instances_train2017.1_1-unlabeled.json'


python tools/test.py configs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_generatepl_1_2.py  work_dirs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_sup_1_2/epoch_24.pth --out labels/unlabeled_1_2_coco.pkl
python scripts/coco/pkl2json.py  labels/unlabeled_1_2_coco.pkl --generatepl_config 'configs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_generatepl_1_2.py'
python scripts/coco/filter_pl.py labels/unlabeled_1_2_coco.pkl --labeled_json 'data/coco/annotations/instances_train2017.1_2-labeled.json' --unlabeled_json 'data/coco/annotations/instances_train2017.1_2-unlabeled.json'
python scripts/coco/form_ann.py  labels/unlabeled_1_2_coco.pkl data/coco/annotations/instances_train2017.1_2-pl.json --labeled_json 'data/coco/annotations/instances_train2017.1_2-labeled.json' --unlabeled_json 'data/coco/annotations/instances_train2017.1_2-unlabeled.json'

python tools/test.py configs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_generatepl_1_4.py  work_dirs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_sup_1_4/epoch_24.pth --out labels/unlabeled_1_4_coco.pkl
python scripts/coco/pkl2json.py  labels/unlabeled_1_4_coco.pkl --generatepl_config 'configs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_generatepl_1_4.py'
python scripts/coco/filter_pl.py labels/unlabeled_1_4_coco.pkl --labeled_json 'data/coco/annotations/instances_train2017.1_4-labeled.json' --unlabeled_json 'data/coco/annotations/instances_train2017.1_4-unlabeled.json'
python scripts/coco/form_ann.py  labels/unlabeled_1_4_coco.pkl data/coco/annotations/instances_train2017.1_4-pl.json --labeled_json 'data/coco/annotations/instances_train2017.1_4-labeled.json' --unlabeled_json 'data/coco/annotations/instances_train2017.1_4-unlabeled.json'

python tools/test.py configs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_generatepl_1_8.py  work_dirs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_sup_1_8/epoch_24.pth --out labels/unlabeled_1_8_coco.pkl
python scripts/coco/pkl2json.py  labels/unlabeled_1_8_coco.pkl --generatepl_config 'configs/noisyboundaries/coco/mask_rcnn_r50_fpn_2x_coco_generatepl_1_8.py'
python scripts/coco/filter_pl.py labels/unlabeled_1_8_coco.pkl --labeled_json 'data/coco/annotations/instances_train2017.1_8-labeled.json' --unlabeled_json 'data/coco/annotations/instances_train2017.1_8-unlabeled.json'
python scripts/coco/form_ann.py  labels/unlabeled_1_8_coco.pkl data/coco/annotations/instances_train2017.1_8-pl.json --labeled_json 'data/coco/annotations/instances_train2017.1_8-labeled.json' --unlabeled_json 'data/coco/annotations/instances_train2017.1_8-unlabeled.json'
